use std::marker::PhantomData;

use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::StorageType,
    prelude::TensorHandleRef,
    server::LaunchError,
    std::{CubeOption, tensor::TensorHandle},
};
use cubek_matmul::{
    components::{
        global::read::{AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading},
        tile::{TileMatmulFamily, io::Strided},
    },
    definition::AvailableLineSizes,
    launch::{TensorArgs, TensorMapArgs},
    routines::specialized::SpecializedAlgorithm,
};

use crate::{
    algorithm::{Algorithm, into_tensor_handle, into_tensor_handle_tma},
    components::{
        ConvolutionOperation,
        global::{args::RuntimeArgs, read::strategy::sync_bias::SyncBiasLoading},
    },
};

/// Cmma convolution
pub struct SpecializedConv<TMM: TileMatmulFamily, L: AsyncPartialLoadingStrategy<RuntimeArgs>> {
    _tmm: PhantomData<TMM>,
    _loader: PhantomData<L>,
}

// pub type SpecializedCyclicConv<TMM> =
//     SpecializedConv<TMM, AsyncPartialCyclicLoading<ColMajorTilingOrder>>;
// pub type SpecializedStridedConv<TMM> = SpecializedConv<TMM, AsyncPartialStridedLoading>;

pub struct SpecializedTmaConv<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<
    TMM: TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    L: AsyncPartialLoadingStrategy<RuntimeArgs, TileKind = Strided>,
> Algorithm for SpecializedConv<TMM, L>
{
    type Routine = SpecializedAlgorithm<TMM, L, SyncBiasLoading>;
    type Args = TensorArgs<RuntimeArgs>;
    const IS_SPECIALIZED: bool = true;

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
> Algorithm for SpecializedTmaConv<TMM>
{
    type Routine = SpecializedAlgorithm<TMM, AsyncPartialTmaLoading, SyncBiasLoading>;
    type Args = TensorMapArgs<RuntimeArgs>;
    const IS_SPECIALIZED: bool = true;

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
