use cubecl::{
    Runtime, client::ComputeClient, ir::StorageType, prelude::TensorBinding, server::LaunchError,
};
use cubek_matmul::{
    components::{
        global::read::{
            AsyncPartialLoadingStrategy, async_partial_cyclic::AsyncPartialCyclicLoading,
            async_partial_strided::AsyncPartialStridedLoading,
            async_partial_tma::AsyncPartialTmaLoading,
        },
        stage::ColMajorTilingOrder,
    },
    definition::{AvailableVectorSizes, TilingBlueprint},
    launch::{TensorArgs, TensorMapArgs},
    routines::specialized::{SpecializedAlgorithm, SpecializedStrategy},
};
use cubek_std::tile::Strided;
use std::marker::PhantomData;

use crate::{
    components::{
        ConvolutionOperation,
        global::{args::RuntimeArgs, read::strategy::sync_bias::SyncBiasLoading},
    },
    routines::{Routine, contiguous_pitched_layout, into_tensor_handle_tma},
};

/// Cmma convolution with a partial async loading strategy.
pub struct SpecializedConv<L: AsyncPartialLoadingStrategy<RuntimeArgs>> {
    _loader: PhantomData<L>,
}

pub type SpecializedAsyncCyclicConv =
    SpecializedConv<AsyncPartialCyclicLoading<ColMajorTilingOrder>>;
pub type SpecializedAsyncStridedConv = SpecializedConv<AsyncPartialStridedLoading>;

pub struct SpecializedTmaConv;

impl<L: AsyncPartialLoadingStrategy<RuntimeArgs, TileKind = Strided>> Routine
    for SpecializedConv<L>
{
    type Blueprint = TilingBlueprint;
    type Strategy = SpecializedStrategy;
    type MatmulRoutine = SpecializedAlgorithm<L, SyncBiasLoading>;
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

impl Routine for SpecializedTmaConv {
    type Blueprint = TilingBlueprint;
    type Strategy = SpecializedStrategy;
    type MatmulRoutine = SpecializedAlgorithm<AsyncPartialTmaLoading, SyncBiasLoading>;
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
