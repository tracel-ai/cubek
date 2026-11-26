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
    /// The extents of `problem` split by `partitioning`, as a routine prepared it: prepare declines a
    /// row of more values than 32 bits count.
    pub fn new(problem: &ReduceRoutineProblem, partitioning: &Partitioning) -> Self {
        let part = partitioning.box_of(ComputeScope::Cube).extent(REDUCED);
        Self {
            part_count: problem.reduced_extent().div_ceil(part) as u32,
            value_count: problem.reduced_extent() as u32,
        }
    }
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

reduce_kernel! {
    /// Every cube's part of its rows, accumulated into the carrier its operation accumulates into;
    /// the last of a box of rows' cubes merges their partials and writes the rows.
    ///
    /// - `partials`: one tensor of the input's shape, the reduced dim holding every part's carrier.
    /// - `partial_indices`: one tensor laid out as `partials`, where the carrier keeps its indices.
    /// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
    /// - `extents`: two scalars, the parts a row splits into and the values a row holds.
    hypercube_last_arrival_reduce_kernel[
        partials: &mut TileTensor<'_>,
        partial_indices: ComptimeOption<&mut TileTensor<'static>>,
        arrivals: &[Atomic<u32>],
        extents: ReduceSplitExtents,
    ]: HypercubeLastArrivalReduceMicrokernel,
    |microkernel, partitioning, cube, x, o, i|
    {
        let load = x.vector_tile();
        let row_lanes = comptime!(load.extent_along(REDUCED));
        let p = partials.tile_mut(&partitioning);
        let pi: ComptimeOption<Tile> =
            partial_indices.map(|partial_indices| partial_indices.tile_mut(&partitioning));
    }
    {
        let place = split_place(&cube, row_lanes);
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N, F::Carrier<Acc, N, K>>(
                &x.at(&cube),
                &mut o.at(&cube),
                &i.clone().map(|i| i.at(&cube)),
                &mut p.at(&cube),
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
    use cubecl::tile::Levels;
    use cubek_microkernel::reduce::ReduceOperation;

    #[test]
    fn a_row_is_split_into_as_many_parts_as_cover_it() {
        const PART_VALUE_COUNT: usize = 128;
        let problem = ReduceRoutineProblem::dense(&[2, 300], 1, ReduceOperation::Sum);
        let levels = Levels::leaf(&problem.split_owner_box(1, PART_VALUE_COUNT))
            .cubes(&[REDUCED, BATCH])
            .build();
        let partitioning = Partitioning::new(problem.space(), levels);
        let extents = ReduceSplitExtents::new(&problem, &partitioning);
        assert_eq!((extents.part_count, extents.value_count), (3, 300));
    }
}
