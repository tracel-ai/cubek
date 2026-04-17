use cubecl::{
    Runtime, client::ComputeClient, ir::StorageType, prelude::TensorBinding, server::LaunchError,
};
use cubek_matmul::{
    components::{
        global::read::{AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading},
        tile::TileMatmulFamily,
    },
    definition::AvailableVectorSizes,
    launch::{TensorArgs, TensorMapArgs},
    routines::specialized::SpecializedAlgorithm,
};
use cubek_std::tile::Strided;
use std::marker::PhantomData;

use crate::{
    algorithm::{Algorithm, contiguous_pitched_layout, into_tensor_handle_tma},
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

impl<TMM: TileMatmulFamily, L: AsyncPartialLoadingStrategy<RuntimeArgs, TileKind = Strided>>
    Algorithm for SpecializedConv<TMM, L>
{
    type Routine = SpecializedAlgorithm<TMM, L, SyncBiasLoading>;
    type Args = TensorArgs<RuntimeArgs>;
    const IS_SPECIALIZED: bool = true;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<TMM: TileMatmulFamily> Algorithm for SpecializedTmaConv<TMM> {
    type Routine = SpecializedAlgorithm<TMM, AsyncPartialTmaLoading, SyncBiasLoading>;
    type Args = TensorMapArgs<RuntimeArgs>;
    const IS_SPECIALIZED: bool = true;

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
