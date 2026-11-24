use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrierFamily;
use cubek_microkernel::reduce::plane::PlaneShuffleReduceMicrokernel;

reduce_kernel! {
    /// Every plane of the launch, reducing its rows into the carrier its operation accumulates into.
    ///
    plane_shuffle_reduce_kernel[]: PlaneShuffleReduceMicrokernel,
    |microkernel, partitioning, plane, x, o, i|
    {}
    {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N, F::Carrier<Acc, N, K>>(&plane, &x, &o, &i);
    }
}
