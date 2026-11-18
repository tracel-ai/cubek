use crate::kernels::ReduceRoutineProblem;
use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Dimension, Partitioning, Region, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceLayout;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;

/// Most row dims a hypercube's cubes run along, beside the reduced dim: a launch grid has three.
pub(crate) const MAX_SPLIT_ROW_DIMS: usize = 2;

/// Where a hypercube's cubes sit, compiled into its kernel: the reduced dimension cut into parts, a
/// cube a part, and the row dimensions, at most two, outermost first, that its boxes of rows lie
/// along.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReduceSplit {
    pub reduced: Dimension,
    pub rows: Vec<Dimension>,
}

impl ReduceSplit {
    /// Where the cubes of a hypercube that `partitioning` places sit: its cube level runs along the
    /// reduced dimension first, then the row dimensions, as the routine states them. The kernel
    /// reads it from its partitioning when it compiles.
    pub fn of(partitioning: &Partitioning) -> Self {
        let cubes = partitioning
            .level_distributed(ComputeScope::Cube)
            .expect("a hypercube's partitioning has a cube level")
            .dimensions();
        let (reduced, rows) = cubes
            .split_first()
            .expect("a hypercube's cubes run along the reduced dimension");
        Self {
            reduced: *reduced,
            rows: rows.to_vec(),
        }
    }
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

impl ReduceSplitExtents {
    /// The extents of `problem` split by `partitioning`, its owners `width` rows across the
    /// reduced dim.
    ///
    /// # Panics
    ///
    /// Where the reduced extent, its parts or the boxes of rows do not fit in 32 bits.
    pub fn new(problem: &ReduceRoutineProblem, partitioning: &Partitioning, width: usize) -> Self {
        let part = partitioning
            .box_of(ComputeScope::Cube)
            .extent(problem.reduced());
        let inner_boxes = problem
            .row_boxes_at_width(width)
            .last()
            .map_or(1, |&(_, boxes)| boxes);
        Self {
            parts: u32_of_extent(problem.reduced_extent().div_ceil(part)),
            values: u32_of_extent(problem.reduced_extent()),
            inner_boxes: u32_of_extent(inner_boxes),
        }
    }

    /// Boxes of rows of `problem`, its owners `width` rows across the reduced dim: each its cubes'
    /// counter.
    pub fn row_boxes(problem: &ReduceRoutineProblem, width: usize) -> usize {
        problem
            .row_boxes_at_width(width)
            .iter()
            .map(|&(_, boxes)| boxes)
            .product()
    }
}

fn u32_of_extent(extent: usize) -> u32 {
    u32::try_from(extent).expect("a reduced extent fits in 32 bits")
}

/// Where `cube` sits along its row: its box of rows' counter, its part, and its part's first line
/// in the row.
#[derive(CubeType)]
struct SplitPlace {
    pub counter: usize,
    pub part: usize,
    pub first_line: usize,
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

/// Every cube's part of its rows, accumulated into a carrier of values alone; the last of a box of
/// rows' cubes merges their partials and writes the rows.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `partials`: one tensor of the input's shape, the reduced dim holding every part's carrier.
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
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    let p = partials.tile(&partitioning);
    for cube in &partitioning {
        let place = split_place::<N>(
            &cube,
            &extents,
            comptime!(ReduceSplit::of(&partitioning.clone())),
            microkernel.part().item().layout,
        );
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
/// - `extents`: three scalars, the parts a row splits into, the values a row holds and the boxes
///   of rows along the inner row dim.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_last_arrival_reduce_indexed_kernel<
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
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
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
            comptime!(ReduceSplit::of(&partitioning.clone())),
            microkernel.part().item().layout,
        );
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
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
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
            comptime!(ReduceSplit::of(&partitioning.clone())),
            microkernel.part().item().layout,
        );
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
