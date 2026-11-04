use cubecl::ir::ElemType;
use cubecl::tile::{Indexed, Partitioning, TileTensor, TopK};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrier;
use cubek_microkernel::reduce::plane::{PlaneReduceMicrokernel, plane_reduce_microkernel};

/// Every plane of the launch, reducing its rows into the carrier its operation accumulates into.
#[cube(launch_unchecked)]
pub fn plane_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    for plane in &partitioning {
        let mut rows_out = o.at(&plane);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&plane);
        let rows = x.at(&plane);
        match comptime!(microkernel.item.operation.carrier()) {
            ReduceCarrier::Values => {
                plane_reduce_microkernel::<In, Acc, Out, Out, N, Vector<Acc, N>>(
                    &rows,
                    &mut rows_out,
                    &mut unwritten,
                    microkernel,
                )
            }
            ReduceCarrier::Indexed => {
                plane_reduce_microkernel::<In, Acc, Out, Out, N, Indexed<Acc, N>>(
                    &rows,
                    &mut rows_out,
                    &mut unwritten,
                    microkernel,
                )
            }
            ReduceCarrier::TopK { k } => {
                let size!(K) = k;
                plane_reduce_microkernel::<In, Acc, Out, Out, N, TopK<Acc, N, K>>(
                    &rows,
                    &mut rows_out,
                    &mut unwritten,
                    microkernel,
                )
            }
        }
    }
}

/// Every plane of the launch, reducing its rows into the carrier its operation accumulates into, and
/// writing the values to `out` and their indices to `indices`, laid out as `out`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn plane_reduce_with_indices_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    indices: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    let i = indices.tile(&partitioning);
    for plane in &partitioning {
        let mut rows_out = o.at(&plane);
        let mut rows_indices = i.at(&plane);
        let rows = x.at(&plane);
        match comptime!(microkernel.item.operation.carrier()) {
            ReduceCarrier::Values => {
                comptime!(panic!("the values themselves carry no index to write"))
            }
            ReduceCarrier::Indexed => {
                plane_reduce_microkernel::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
                    &rows,
                    &mut rows_out,
                    &mut rows_indices,
                    microkernel,
                )
            }
            ReduceCarrier::TopK { k } => {
                let size!(K) = k;
                plane_reduce_microkernel::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(
                    &rows,
                    &mut rows_out,
                    &mut rows_indices,
                    microkernel,
                )
            }
        }
    }
}
