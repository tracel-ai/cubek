use cubecl::{
    server::LaunchError,
    {Runtime, client::ComputeClient, ir::StorageType, prelude::TensorBinding},
};
use cubek_matmul::components::{
    global::read::FullLoadingStrategy,
    tile::{StandardTileIO, TileMatmulFamily},
};
use cubek_matmul::components::{
    global::read::sync_full_cyclic::SyncFullCyclicLoading,
    stage::{ColMajorTilingOrder, RowMajorTilingOrder},
};
use cubek_matmul::{
    components::global::read::{
        async_full_tma::AsyncFullTmaLoading, sync_full_strided::SyncFullStridedLoading,
        sync_full_tilewise::SyncFullTilewiseLoading,
    },
    routines::simple::SimpleAlgorithm,
};
use cubek_matmul::{
    definition::AvailableVectorSizes,
    launch::{TensorArgs, TensorMapArgs},
};
use cubek_std::tile::Strided;
use std::marker::PhantomData;

use crate::{
    algorithm::{contiguous_pitched_layout, into_tensor_handle_tma},
    components::{
        ConvolutionOperation,
        global::{
            args::RuntimeArgs,
            read::strategy::{
                async_full_cyclic::AsyncFullCyclicLoading,
                async_full_strided::AsyncFullStridedLoading, sync_bias::SyncBiasLoading,
            },
        },
    },
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConv<
    TMM: TileMatmulFamily,
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
> {
    _tmm: PhantomData<TMM>,
    _loader: PhantomData<(LL, LR)>,
}

pub type SimpleSyncCyclicConv<TMM> = SimpleConv<
    TMM,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
>;
pub type SimpleSyncStridedConv<TMM> =
    SimpleConv<TMM, SyncFullStridedLoading, SyncFullStridedLoading>;
pub type SimpleSyncTilewiseConv<TMM> = SimpleConv<
    TMM,
    SyncFullTilewiseLoading<RowMajorTilingOrder>,
    SyncFullTilewiseLoading<ColMajorTilingOrder>,
>;
pub type SimpleAsyncCyclicConv<TMM> = SimpleConv<
    TMM,
    AsyncFullCyclicLoading<RowMajorTilingOrder>,
    AsyncFullCyclicLoading<ColMajorTilingOrder>,
>;
pub type SimpleAsyncStridedConv<TMM> =
    SimpleConv<TMM, AsyncFullStridedLoading, AsyncFullStridedLoading>;

pub struct SimpleAsyncTmaConv<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<
    TMM: TileMatmulFamily<TileIO = StandardTileIO>,
    LL: FullLoadingStrategy<RuntimeArgs, TileKind = Strided>,
    LR: FullLoadingStrategy<RuntimeArgs, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
> Algorithm for SimpleConv<TMM, LL, LR>
{
    type Routine = SimpleAlgorithm<TMM, LL, LR, SyncBiasLoading>;
    type Args = TensorArgs<RuntimeArgs>;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<TMM: TileMatmulFamily<TileIO = StandardTileIO>> Algorithm for SimpleAsyncTmaConv<TMM> {
    type Routine = SimpleAlgorithm<TMM, AsyncFullTmaLoading, AsyncFullTmaLoading, SyncBiasLoading>;

    type Args = TensorMapArgs<RuntimeArgs>;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
        into_tensor_handle_tma(client, handle, dtype, operation)
    }

    fn filter_vector_sizes(vector_sizes: AvailableVectorSizes) -> AvailableVectorSizes {
        AvailableVectorSizes {
            lhs: vec![1],
            rhs: vec![1],
            out: vector_sizes.out,
        }
    }
}
