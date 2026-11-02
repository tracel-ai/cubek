use cubecl::ir::ElemType;
use cubecl::tile::{Indexed, Partitioning, TileArg, TopK};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrier;
use cubek_microkernel::reduce::unit::{UnitReduceMicrokernel, unit_reduce_microkernel};

/// Every unit of the launch, reducing its rows into the carrier its operation folds into.
#[cube(launch_unchecked)]
pub fn unit_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileArg<'_>,
    out: &TileArg<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    for unit in &partitioning {
        let mut rows_out = o.at(&unit);
        let rows = x.at(&unit);
        match comptime!(microkernel.item.operation.carrier()) {
            ReduceCarrier::Values => unit_reduce_microkernel::<In, Acc, Out, N, Vector<Acc, N>>(
                &rows,
                &mut rows_out,
                microkernel,
            ),
            ReduceCarrier::Indexed => unit_reduce_microkernel::<In, Acc, Out, N, Indexed<Acc, N>>(
                &rows,
                &mut rows_out,
                microkernel,
            ),
            ReduceCarrier::TopK { k } => {
                let size!(K) = k;
                unit_reduce_microkernel::<In, Acc, Out, N, TopK<Acc, N, K>>(
                    &rows,
                    &mut rows_out,
                    microkernel,
                )
            }
        }
    }
}
