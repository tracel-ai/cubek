use cubecl::{
    Runtime, client::ComputeClient, ir::StorageType, prelude::TensorBinding, server::LaunchError,
};
use cubek_matmul::{
    components::global::read::{
        AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading,
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
pub struct SpecializedConv<L: AsyncPartialLoadingStrategy<RuntimeArgs>> {
    _loader: PhantomData<L>,
}

pub struct SpecializedTmaConv;

impl<L: AsyncPartialLoadingStrategy<RuntimeArgs, TileKind = Strided>> Algorithm
    for SpecializedConv<L>
{
    type Routine = SpecializedAlgorithm<L, SyncBiasLoading>;
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

impl Algorithm for SpecializedTmaConv {
    type Routine = SpecializedAlgorithm<AsyncPartialTmaLoading, SyncBiasLoading>;
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
