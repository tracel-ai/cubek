use cubecl::{
    server::LaunchError,
    {Runtime, client::ComputeClient, ir::StorageType, prelude::TensorBinding},
};
use cubek_matmul::components::global::read::FullLoadingStrategy;
use cubek_matmul::components::{
    global::read::sync_full_cyclic::SyncFullCyclicLoading,
    stage::{ColMajorTilingOrder, RowMajorTilingOrder},
};
use cubek_matmul::{
    components::global::read::{
        async_full_tma::AsyncFullTmaLoading, sync_full_strided::SyncFullStridedLoading,
        sync_full_tilewise::SyncFullTilewiseLoading,
    },
    routines::simple::{SimpleAlgorithm, SimpleArgs},
};
use cubek_matmul::{
    definition::{AvailableVectorSizes, TilingBlueprint},
    launch::{TensorArgs, TensorMapArgs},
};
use cubek_std::tile::Strided;
use std::marker::PhantomData;

use crate::{
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
    routines::{Routine, contiguous_pitched_layout, into_tensor_handle_tma},
};

/// Cmma convolution
pub struct SimpleConv<LL: FullLoadingStrategy<RuntimeArgs>, LR: FullLoadingStrategy<RuntimeArgs>> {
    _loader: PhantomData<(LL, LR)>,
}

pub type SimpleSyncCyclicConv = SimpleConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
>;
pub type SimpleSyncStridedConv = SimpleConv<SyncFullStridedLoading, SyncFullStridedLoading>;
pub type SimpleSyncTilewiseConv = SimpleConv<
    SyncFullTilewiseLoading<RowMajorTilingOrder>,
    SyncFullTilewiseLoading<ColMajorTilingOrder>,
>;
pub type SimpleAsyncCyclicConv = SimpleConv<
    AsyncFullCyclicLoading<RowMajorTilingOrder>,
    AsyncFullCyclicLoading<ColMajorTilingOrder>,
>;
pub type SimpleAsyncStridedConv = SimpleConv<AsyncFullStridedLoading, AsyncFullStridedLoading>;

pub struct SimpleAsyncTmaConv;

impl<
    LL: FullLoadingStrategy<RuntimeArgs, TileKind = Strided>,
    LR: FullLoadingStrategy<RuntimeArgs, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
> Routine for SimpleConv<LL, LR>
{
    type Blueprint = TilingBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine = SimpleAlgorithm<LL, LR, SyncBiasLoading>;
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

impl Routine for SimpleAsyncTmaConv {
    type Blueprint = TilingBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine = SimpleAlgorithm<AsyncFullTmaLoading, AsyncFullTmaLoading, SyncBiasLoading>;
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
