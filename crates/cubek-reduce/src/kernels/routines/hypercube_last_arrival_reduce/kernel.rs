use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Dimension, Partitioning, Region, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceLayout;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;

/// Where a hypercube's cubes sit, compiled into its kernel: the reduced dimension cut into parts, a
/// cube a part, and the row dimensions, at most two, outermost first, that its boxes of rows lie
/// along.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReduceSplit {
    pub reduced: Dimension,
    pub rows: Vec<Dimension>,
}

/// What a hypercube launch compiles beside its partitioning, crossing the launch as one argument:
/// the microkernel each cube runs, and where the cubes sit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalReduceDecisions {
    pub microkernel: HypercubeLastArrivalReduceMicrokernel,
    pub split: ReduceSplit,
}

/// The extents a hypercube launch is handed at run time, which its kernel does not compile.
#[derive(CubeType, CubeLaunch, Debug, Clone, Copy)]
pub struct ReduceSplitExtents {
    /// Parts a row is split into, a cube each.
    pub parts: u32,
    /// Values a row holds, which a mean divides by.
    pub values: u32,
    /// Boxes of rows along the inner row dimension, where there are two.
    pub inner_boxes: u32,
}

/// Where `cube` sits along its row: its box of rows' counter, its part, and its part's first line
/// in the row.
#[derive(CubeType)]
struct SplitPlace {
    counter: usize,
    part: usize,
    first_line: usize,
}

#[cube]
fn split_place<N: Size>(
    cube: &Region,
    extents: &ReduceSplitExtents,
    #[comptime] split: ReduceSplit,
    #[comptime] layout: ReduceLayout,
) -> SplitPlace {
    let counter = match split.rows.len() {
        0 => 0usize,
        1 => cube.coord(comptime!(split.rows[0])),
        _ => {
            cube.coord(comptime!(split.rows[0])) * extents.inner_boxes as usize
                + cube.coord(comptime!(split.rows[1]))
        }
    };
    let first_line = match layout {
        // Along, a line holds `N` values of the row.
        ReduceLayout::Along => cube.origin(split.reduced) / N::value(),
        ReduceLayout::Across => cube.origin(split.reduced),
    };
    SplitPlace {
        counter,
        part: cube.coord(split.reduced),
        first_line,
    }
}

/// Every cube's part of its rows, accumulated into the carrier its operation accumulates into; the last of
/// a box of rows' cubes merges their partials and writes the rows.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `partials`, `partial_indices`: one tensor each, the input's shape with the reduced dim holding
///   every part's carrier.
/// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
/// - `extents`: three scalars, the parts a row splits into, the values a row holds and the boxes
///   of rows along the inner row dim.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_last_arrival_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &TileTensor<'_>,
    partials: &TileTensor<'_>,
    partial_indices: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] decisions: HypercubeLastArrivalReduceDecisions,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    let p = partials.tile(&partitioning);
    let pi = partial_indices.tile(&partitioning);
    for cube in &partitioning {
        let place = split_place::<N>(
            &cube,
            &extents,
            decisions.split.clone(),
            decisions.microkernel.part().item().layout,
        );
        let mut rows_out = o.at(&cube);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let rows = x.at(&cube);
        let mut slots = p.at(&cube);
        let mut slot_indices = pi.at(&cube);
        decisions
            .microkernel
            .runtime()
            .execute::<In, Acc, Out, Out, N>(
                &rows,
                &mut rows_out,
                &mut unwritten,
                &mut slots,
                &mut slot_indices,
                arrivals,
                place.counter,
                place.part,
                extents.parts as usize,
                place.first_line,
                extents.values as usize,
            );
    }
}

/// Every cube's part of its rows, as [`hypercube_last_arrival_reduce_kernel()`] reduces it, the values written to
/// `out` and their indices to `indices`, laid out as `out`.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor laid out as `out`.
/// - `partials`, `partial_indices`: one tensor each, the input's shape with the reduced dim holding
///   every part's carrier.
/// - `arrivals`: one buffer of `u32` counters, one a box of rows, at zero.
/// - `extents`: three scalars, the parts a row splits into, the values a row holds and the boxes
///   of rows along the inner row dim.
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
    out: &TileTensor<'_>,
    indices: &TileTensor<'_>,
    partials: &TileTensor<'_>,
    partial_indices: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] decisions: HypercubeLastArrivalReduceDecisions,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    let i = indices.tile(&partitioning);
    let p = partials.tile(&partitioning);
    let pi = partial_indices.tile(&partitioning);
    for cube in &partitioning {
        let place = split_place::<N>(
            &cube,
            &extents,
            decisions.split.clone(),
            decisions.microkernel.part().item().layout,
        );
        let mut rows_out = o.at(&cube);
        let mut rows_indices = i.at(&cube);
        let rows = x.at(&cube);
        let mut slots = p.at(&cube);
        let mut slot_indices = pi.at(&cube);
        decisions
            .microkernel
            .runtime()
            .execute::<In, Acc, Out, Idx, N>(
                &rows,
                &mut rows_out,
                &mut rows_indices,
                &mut slots,
                &mut slot_indices,
                arrivals,
                place.counter,
                place.part,
                extents.parts as usize,
                place.first_line,
                extents.values as usize,
            );
    }
}
