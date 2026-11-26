use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrierFamily;
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernel;

reduce_kernel! {
    /// Every cube of the launch, reducing its rows into the carrier its operation accumulates into.
    ///
    cube_staged_reduce_kernel[]: CubeStagedReduceMicrokernel,
    |microkernel, partitioning, cube, x, o, i|
    {}
    {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N, F::Carrier<Acc, N, K>>(&cube, &x, &o, &i);
    }
}
