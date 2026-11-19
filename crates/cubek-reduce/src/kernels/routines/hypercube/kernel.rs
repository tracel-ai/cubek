use crate::kernels::{BATCH, REDUCED};
use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, Region, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceLayout;
use cubek_microkernel::reduce::hypercube::{
    HypercubeReduceMicrokernel, hypercube_reduce_microkernel,
};

/// The extents a hypercube launch is handed at run time, which its kernel does not compile.
#[derive(CubeType, CubeLaunch, Debug, Clone, Copy)]
pub struct ReduceSplitExtents {
    /// Parts a row is split into, a cube each.
    pub parts: u32,
    /// Values a row holds, which a mean divides by.
    pub values: u32,
}

/// Where `cube` sits along its row: its box of rows' counter, its part, and its part's first line
/// in the row. A box of rows' cubes share its [`BATCH`] coordinate, which counts them in.
#[derive(CubeType)]
struct SplitPlace {
    counter: usize,
    part: usize,
    first_line: usize,
}

#[cube]
fn split_place<N: Size>(cube: &Region, #[comptime] layout: ReduceLayout) -> SplitPlace {
    let first_line = match layout {
        // Along, a line holds `N` values of the row.
        ReduceLayout::Along => cube.origin(REDUCED) / N::value(),
        ReduceLayout::Across => cube.origin(REDUCED),
    };
    SplitPlace {
        counter: cube.coord(BATCH),
        part: cube.coord(REDUCED),
        first_line,
    }
}

/// Every cube's part of its rows, accumulated into the carrier its operation accumulates into; the
/// last of a box of rows' cubes merges their partials and writes the rows.
#[cube(launch_unchecked)]
pub fn hypercube_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partials: &TileTensor<'_>,
    partial_indices: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    let p = partials.tile(&partitioning);
    let pi = partial_indices.tile(&partitioning);
    for cube in &partitioning {
        let place = split_place::<N>(&cube, microkernel.item.layout);
        let mut rows_out = o.at(&cube);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let rows = x.at(&cube);
        let mut slots = p.at(&cube);
        let mut slot_indices = pi.at(&cube);
        hypercube_reduce_microkernel::<In, Acc, Out, Out, N>(
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
            microkernel,
        );
    }
}

/// Every cube's part of its rows, as [`hypercube_reduce_kernel()`] reduces it, the values written
/// to `out` and their indices to `indices`, laid out as `out`.
#[cube(launch_unchecked)]
pub fn hypercube_reduce_with_indices_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    indices: &TileTensor<'_>,
    partials: &TileTensor<'_>,
    partial_indices: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    extents: ReduceSplitExtents,
    #[comptime] microkernel: HypercubeReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    let i = indices.tile(&partitioning);
    let p = partials.tile(&partitioning);
    let pi = partial_indices.tile(&partitioning);
    for cube in &partitioning {
        let place = split_place::<N>(&cube, microkernel.item.layout);
        let mut rows_out = o.at(&cube);
        let mut rows_indices = i.at(&cube);
        let rows = x.at(&cube);
        let mut slots = p.at(&cube);
        let mut slot_indices = pi.at(&cube);
        hypercube_reduce_microkernel::<In, Acc, Out, Idx, N>(
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
            microkernel,
        );
    }
}
