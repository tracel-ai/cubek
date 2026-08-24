use cubecl::{
    Runtime, client::ComputeClient, ir::ElemType, prelude::TensorBinding, server::LaunchError,
};
use cubek_matmul::definition::AvailableVectorSizes;
use cubek_matmul::multi_level::args::{TensorArgs, TensorMapArgs};
use cubek_matmul::multi_level::components::global::read::AsyncPartialLoadingStrategy;
use cubek_matmul::multi_level::components::global::read::async_partial_cyclic::AsyncPartialCyclicLoading;
use cubek_matmul::multi_level::components::global::read::async_partial_strided::AsyncPartialStridedLoading;
use cubek_matmul::multi_level::components::global::read::async_partial_tma::AsyncPartialTmaLoading;
use cubek_matmul::multi_level::definition::BatchMatmulBlueprint;
use cubek_matmul::multi_level::routines::batch::specialized::{
    SpecializedAlgorithm, SpecializedStrategy,
};
use cubek_std::tile::ColMajorTilingOrder;
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

impl<L: AsyncPartialLoadingStrategy<RuntimeArgs>> Routine for SpecializedConv<L> {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SpecializedStrategy;
    type MatmulRoutine = SpecializedAlgorithm<L, SyncBiasLoading>;
    type Args = TensorArgs<RuntimeArgs>;
    const IS_SPECIALIZED: bool = true;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl Routine for SpecializedTmaConv {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SpecializedStrategy;
    type MatmulRoutine = SpecializedAlgorithm<AsyncPartialTmaLoading, SyncBiasLoading>;
    type Args = TensorMapArgs<RuntimeArgs>;
    const IS_SPECIALIZED: bool = true;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: ElemType,
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
