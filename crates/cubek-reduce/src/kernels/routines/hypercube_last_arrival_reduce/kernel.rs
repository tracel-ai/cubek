use crate::kernels::{BATCH, REDUCED, ReduceRoutineProblem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Partitioning, Region, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_microkernel::reduce::row_lanes;

/// The extents a hypercube launch is handed at run time, which its kernel does not compile.
#[derive(CubeType, CubeLaunch, Debug, Clone, Copy)]
pub(crate) struct ReduceSplitExtents {
    /// Parts a row is split into, a cube each.
    pub parts: u32,
    /// Values a row holds, which a mean divides by.
    pub values: u32,
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
            parts: u32_of_extent(problem.reduced_extent().div_ceil(part)),
            values: u32_of_extent(problem.reduced_extent()),
        }
    }
}

fn u32_of_extent(extent: usize) -> u32 {
    u32::try_from(extent).expect("a reduced extent fits in 32 bits")
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

/// Every cube's part of its rows, accumulated into a carrier of values alone; the last of a box of
/// rows' cubes merges their partials and writes the rows.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `partials`: one tensor of the input's shape, the reduced dim holding every part's carrier.
/// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
/// - `extents`: two scalars, the parts a row splits into and the values a row holds.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_last_arrival_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    partials: &mut TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    let p = partials.tile_mut(&partitioning);
    for cube in &partitioning {
        let place = split_place(&cube, row_lanes(&x));
        let rows = x.at(&cube);
        let mut rows_out = o.at(&cube);
        let mut slots = p.at(&cube);
        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &rows,
            &mut rows_out,
            &ComptimeOption::new_None(),
            &mut slots,
            &ComptimeOption::new_None(),
            arrivals,
            place.counter,
            place.part,
            extents.parts as usize,
            place.first_line,
            extents.values as usize,
        );
    }
}

/// Every cube's part of its rows, as [`hypercube_last_arrival_reduce_kernel()`] reduces it, into a
/// carrier of values and their indices, whose partials keep both: an argmax, an argmin or a top-k
/// writing one half.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `partials`, `partial_indices`: one tensor each, the input's shape with the reduced dim holding
///   every part's carrier.
/// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
/// - `extents`: two scalars, the parts a row splits into and the values a row holds.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_last_arrival_reduce_indexed_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    partials: &mut TileTensor<'_>,
    partial_indices: &mut TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    let p = partials.tile_mut(&partitioning);
    let pi = partial_indices.tile_mut(&partitioning);
    for cube in &partitioning {
        let place = split_place(&cube, row_lanes(&x));
        let rows = x.at(&cube);
        let mut rows_out = o.at(&cube);
        let mut slots = p.at(&cube);
        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &rows,
            &mut rows_out,
            &ComptimeOption::new_None(),
            &mut slots,
            &ComptimeOption::new_Some(pi.at(&cube)),
            arrivals,
            place.counter,
            place.part,
            extents.parts as usize,
            place.first_line,
            extents.values as usize,
        );
    }
}

/// Every cube's part of its rows, as [`hypercube_last_arrival_reduce_indexed_kernel()`] reduces
/// it, the values written to `out` and their indices to `indices`, laid out as `out`.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor laid out as `out`.
/// - `partials`, `partial_indices`: one tensor each, the input's shape with the reduced dim holding
///   every part's carrier.
/// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
/// - `extents`: two scalars, the parts a row splits into and the values a row holds.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_last_arrival_reduce_with_indices_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    indices: &mut TileTensor<'_>,
    partials: &mut TileTensor<'_>,
    partial_indices: &mut TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    let i = indices.tile_mut(&partitioning);
    let p = partials.tile_mut(&partitioning);
    let pi = partial_indices.tile_mut(&partitioning);
    for cube in &partitioning {
        let place = split_place(&cube, row_lanes(&x));
        let rows = x.at(&cube);
        let mut rows_out = o.at(&cube);
        let mut slots = p.at(&cube);
        microkernel.runtime().execute::<In, Acc, Out, Idx, N>(
            &rows,
            &mut rows_out,
            &ComptimeOption::new_Some(i.at(&cube)),
            &mut slots,
            &ComptimeOption::new_Some(pi.at(&cube)),
            arrivals,
            place.counter,
            place.part,
            extents.parts as usize,
            place.first_line,
            extents.values as usize,
        );
    }
}
