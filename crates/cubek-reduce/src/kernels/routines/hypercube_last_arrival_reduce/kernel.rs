use crate::kernels::{BATCH, REDUCED, ReduceRoutineProblem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Partitioning, Region, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrierFamily;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;

/// The extents a hypercube launch is handed at run time, which its kernel does not compile.
#[derive(CubeType, CubeLaunch, Debug, Clone, Copy)]
pub(crate) struct ReduceSplitExtents {
    /// Parts a row is split into, a cube each.
    pub part_count: u32,
    /// Values a row holds, which a mean divides by.
    pub value_count: u32,
}

impl ReduceSplitExtents {
    /// The extents of `problem` split by `partitioning`.
    ///
    /// # Panics
    ///
    /// Where the reduced extent or its parts do not fit in 32 bits.
    pub fn new(problem: &ReduceRoutineProblem, partitioning: &Partitioning) -> Self {
        let part = partitioning.box_of(ComputeScope::Cube).extent(REDUCED);
        Self {
            part_count: u32_of_value_count(problem.reduced_extent().div_ceil(part)),
            value_count: u32_of_value_count(problem.reduced_extent()),
        }
    }
}

fn u32_of_value_count(value_count: usize) -> u32 {
    u32::try_from(value_count).expect("a row's values count in 32 bits")
}

/// Where `cube` sits along its row: its box of rows' counter, its part, and its part's first line
/// in the row, its lines `row_lanes` values of the row. A box of rows' cubes share its [`BATCH`]
/// coordinate, which counts them in.
#[derive(CubeType)]
struct SplitPlace {
    pub counter: usize,
    pub part: usize,
    pub first_line: usize,
}

#[cube]
fn split_place(cube: &Region, #[comptime] row_lanes: usize) -> SplitPlace {
    SplitPlace {
        counter: cube.coord(BATCH),
        part: cube.coord(REDUCED),
        first_line: cube.origin(REDUCED) / row_lanes,
    }
}

/// Every cube's part of its rows, accumulated into the carrier its operation accumulates into; the
/// last of a box of rows' cubes merges their partials and writes the rows, and their indices beside
/// them where the launch binds any.
///
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor of `out`'s shape at its own strides, where the launch writes the indices.
/// - `partials`: one tensor of the input's shape, the reduced dim holding every part's carrier.
/// - `partial_indices`: one tensor laid out as `partials`, where the carrier keeps its indices.
/// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
/// - `extents`: two scalars, the parts a row splits into and the values a row holds.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_last_arrival_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    indices: ComptimeOption<&mut TileTensor<'static>>,
    partials: &mut TileTensor<'_>,
    partial_indices: ComptimeOption<&mut TileTensor<'static>>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let load = x.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let o = out.tile_mut(&partitioning);
    let i: ComptimeOption<Tile> = indices.map(|indices| indices.tile_mut(&partitioning));
    let p = partials.tile_mut(&partitioning);
    let pi: ComptimeOption<Tile> =
        partial_indices.map(|partial_indices| partial_indices.tile_mut(&partitioning));
    for cube in &partitioning {
        let place = split_place(&cube, row_lanes);
        let rows = x.at(&cube);
        let mut rows_out = o.at(&cube);
        let mut slots = p.at(&cube);
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N, F::Carrier<Acc, N, K>>(
                &rows,
                &mut rows_out,
                &i.clone().map(|i| i.at(&cube)),
                &mut slots,
                &pi.clone().map(|pi| pi.at(&cube)),
                arrivals,
                place.counter,
                place.part,
                extents.part_count as usize,
                place.first_line,
                extents.value_count as usize,
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_of_32_bits_is_kept() {
        assert_eq!(u32_of_value_count(u32::MAX as usize), u32::MAX);
    }

    #[test]
    #[should_panic(expected = "a row's values count in 32 bits")]
    fn a_count_past_32_bits_panics() {
        u32_of_value_count(u32::MAX as usize + 1);
    }
}
