use cubecl::server::LaunchError;
use cubecl::std::{CubeOption, tensor::TensorHandle};
use cubecl::{Runtime, client::ComputeClient, ir::StorageType, prelude::TensorHandleRef};
use cubek_matmul::components::stage::PlaneMatmulFamily;
use cubek_matmul::components::tile::TileMatmulFamily;
use cubek_matmul::components::{
    global::read::sync_full_cyclic::SyncFullCyclicLoading,
    stage::{ColMajorTilingOrder, RowMajorTilingOrder},
};
use cubek_matmul::components::{
    global::read::{
        async_full_tma::AsyncFullTmaLoading, sync_full_strided::SyncFullStridedLoading,
        sync_full_tilewise::SyncFullTilewiseLoading,
    },
    stage::StridedStageFamily,
    tile::io::Strided,
};
use cubek_matmul::definition::{MatmulElems, MatmulLineSizes, MatmulSetupError, TilingBlueprint};
use cubek_matmul::launch::{TensorArgs, TensorMapArgs};
use std::marker::PhantomData;

use crate::{
    components::{
        ConvolutionOperation, ConvolutionProblem, convolution_matmul_selection,
        global::{
            read::{
                full_reader::FullLoadingStrategy,
                strategy::{
                    async_full_cyclic::AsyncFullCyclicLoading,
                    async_full_strided::AsyncFullStridedLoading,
                },
            },
            single_stage::simple::SimpleConvolutionFamily,
        },
    },
    kernels::forward::{into_tensor_handle, into_tensor_handle_tma},
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConv<TMM: TileMatmulFamily, LL: FullLoadingStrategy, LR: FullLoadingStrategy> {
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
    LL: FullLoadingStrategy,
    LR: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
> Algorithm for SimpleConv<TMM, LL, LR>
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        Option<StridedStageFamily>,
    >;
    type GlobalConvolution = SimpleConvolutionFamily<Self::StageMatmul, LL, LR>;

    type Args = TensorArgs;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorHandle<R>, LaunchError> {
        into_tensor_handle(client, handle, dtype)
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        dtypes: &mut MatmulElems,
    ) -> Result<TilingBlueprint, MatmulSetupError> {
        Ok(convolution_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            TMM::should_swizzle(client),
            line_sizes,
            dtypes,
        )?)
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
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        Option<StridedStageFamily>,
    >;
    type GlobalConvolution =
        SimpleConvolutionFamily<Self::StageMatmul, AsyncFullTmaLoading, AsyncFullTmaLoading>;

    type Args = TensorMapArgs;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorHandle<R>, LaunchError> {
        into_tensor_handle_tma(client, handle, dtype, operation)
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        dtypes: &mut MatmulElems,
    ) -> Result<TilingBlueprint, MatmulSetupError> {
        if line_sizes.lhs > 1 || line_sizes.rhs > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Not available with input line sizes > 1",
            )));
        }

        Ok(convolution_matmul_selection::<TMM, R>(
            client, problem, plane_dim, false, line_sizes, dtypes,
        )?)
    }
}
