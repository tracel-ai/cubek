use cubecl::server::LaunchError;
use cubecl::std::{CubeOption, tensor::TensorHandle};
use cubecl::{Runtime, client::ComputeClient, ir::StorageType, prelude::TensorHandleRef};
use cubek_matmul::components::{global::read::FullLoadingStrategy, tile::TileMatmulFamily};
use cubek_matmul::components::{
    global::read::sync_full_cyclic::SyncFullCyclicLoading,
    stage::{ColMajorTilingOrder, RowMajorTilingOrder},
};
use cubek_matmul::definition::AvailableLineSizes;
use cubek_matmul::launch::{TensorArgs, TensorMapArgs};
use cubek_matmul::{
    components::{
        global::read::{
            async_full_tma::AsyncFullTmaLoading, sync_full_strided::SyncFullStridedLoading,
            sync_full_tilewise::SyncFullTilewiseLoading,
        },
        tile::io::Strided,
    },
    routines::simple::SimpleAlgorithm,
};
use std::marker::PhantomData;

use crate::{
    algorithm::{into_tensor_handle, into_tensor_handle_tma},
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
    TMM: TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    LL: FullLoadingStrategy<RuntimeArgs, TileKind = Strided>,
    LR: FullLoadingStrategy<RuntimeArgs, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
> Algorithm for SimpleConv<TMM, LL, LR>
{
    type Routine = SimpleAlgorithm<TMM, LL, LR, SyncBiasLoading>;
    type Args = TensorArgs<RuntimeArgs>;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorHandle<R>, LaunchError> {
        into_tensor_handle(client, handle, dtype)
    }
}

impl<
    TMM: TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
> Algorithm for SimpleAsyncTmaConv<TMM>
{
    type Routine = SimpleAlgorithm<TMM, AsyncFullTmaLoading, AsyncFullTmaLoading, SyncBiasLoading>;

    type Args = TensorMapArgs<RuntimeArgs>;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorHandle<R>, LaunchError> {
        into_tensor_handle_tma(client, handle, dtype, operation)
    }

    fn filter_line_sizes(line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        AvailableLineSizes {
            lhs: vec![1],
            rhs: vec![1],
            out: line_sizes.out,
        }
    }
}
