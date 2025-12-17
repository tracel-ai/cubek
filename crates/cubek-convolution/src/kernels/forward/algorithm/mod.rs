use cubek_matmul::definition::{
    LoadingPrecomputeStrategy, MatmulElems, MatmulLineSizes, MatmulSetupError, MultiRowStrategy,
    TilingBlueprint,
};
use cubek_matmul::{
    components::{
        global::{LoadSpecializationConfig, read::ReaderMode},
        stage::{PartitionBuffering, StageMatmulFamily},
        tile::TileMatmulFamily,
    },
    launch::MatmulArgs,
};

use cubecl::std::tensor::{TensorHandle, into_contiguous_pitched, is_contiguous_pitched};

use cubecl::prelude::*;

use crate::components::{
    ConvolutionOperation, ConvolutionProblem,
    global::{GlobalConfig, GlobalConvolutionFamily},
};

pub mod simple;

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily;
    type GlobalConvolution: GlobalConvolutionFamily;

    type Args: MatmulArgs;

    fn cube_count(selection: &TilingBlueprint, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme.elements_per_stage_along_m();
        let n_stage = selection.tiling_scheme.elements_per_stage_along_n();
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn multi_row_strategy() -> MultiRowStrategy {
        MultiRowStrategy::Never
    }

    fn loading_precompute_strategy() -> LoadingPrecomputeStrategy {
        LoadingPrecomputeStrategy::Never
    }

    fn reader_mode() -> ReaderMode {
        ReaderMode::Relaxed
    }

    fn load_specialization() -> LoadSpecializationConfig {
        LoadSpecializationConfig::default()
    }

    fn partition_buffering_strategy() -> PartitionBuffering {
        PartitionBuffering::Double
    }

    /// Make a convolution config from a convolution problem, and launch options
    fn expand_config<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        selection: &TilingBlueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<GlobalConfig<Self::GlobalConvolution>, MatmulSetupError> {
        Self::GlobalConvolution::expand_config(client, problem, selection, line_sizes, dtypes)
    }

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorHandle<R>, LaunchError>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        matmul_elems: &mut MatmulElems,
    ) -> Result<TilingBlueprint, MatmulSetupError>;
}

pub(crate) fn into_tensor_handle<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let handle = if has_valid_layout(handle) {
        TensorHandle::from_ref(handle, dtype)
    } else {
        into_contiguous_pitched(client, handle, dtype)?
    };
    Ok(handle)
}

fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
    let rank = handle.shape.len();
    let dim_c = rank - 1;
    handle.strides[dim_c] == 1
}

const TMA_STRIDE_ALIGN: usize = 16;

pub(crate) fn into_tensor_handle_tma<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TensorHandleRef<'_, R>,
    dtype: StorageType,
    operation: ConvolutionOperation,
) -> Result<TensorHandle<R>, LaunchError> {
    let handle = if has_valid_layout_tma(handle, operation) {
        TensorHandle::from_ref(handle, dtype)
    } else {
        into_contiguous_pitched(client, handle, dtype)?
    };
    Ok(handle)
}

pub(crate) fn has_valid_layout_tma<R: Runtime>(
    handle: &TensorHandleRef<'_, R>,
    operation: ConvolutionOperation,
) -> bool {
    let stride_align = TMA_STRIDE_ALIGN / handle.elem_size;
    let rank = handle.shape.len();
    let dim_c = rank - 1;

    let aligned = handle.strides[..dim_c]
        .iter()
        .all(|stride| stride % stride_align == 0);

    let valid_layout = handle.strides[dim_c] == 1;

    let is_valid_wgrad = if operation == ConvolutionOperation::BackwardWeight {
        is_contiguous_pitched(handle.shape, handle.strides)
    } else {
        true
    };

    valid_layout && aligned && is_valid_wgrad
}
