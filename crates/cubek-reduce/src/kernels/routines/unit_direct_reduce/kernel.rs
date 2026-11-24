use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrierFamily;
use cubek_microkernel::reduce::unit::UnitDirectReduceMicrokernel;

reduce_kernel! {
    /// Every unit of the launch, reducing its rows into the carrier its operation accumulates into.
    ///
    unit_direct_reduce_kernel[]: UnitDirectReduceMicrokernel,
    |microkernel, partitioning, unit, x, o, i|
    {}
    {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N, F::Carrier<Acc, N, K>>(
                &x.at(&unit),
                &mut o.at(&unit),
                &i.clone().map(|i| i.at(&unit)),
            );
    }
}
