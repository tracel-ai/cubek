use cubecl::prelude::*;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubek_std::InputBinding;
use std::fmt::{Debug, Display};

use crate::definition::{MatmulSetupError, MatmulVectorSizes};

/// A launch-time config carried alongside the tensors. `()` for a plain matmul; the
/// fusion path substitutes its own.
pub trait RuntimeConfig: LaunchArg + CubeType<ExpandType: Clone> + Clone + Send + Sync {}
impl<T: LaunchArg + CubeType<ExpandType: Clone> + Clone + Send + Sync> RuntimeConfig for T {}

/// A stride-0 (broadcast) matrix dim, or any layout that isn't row/col-major,
/// classifies as [`MatrixBatchLayout::HighlyPermuted`] and can't be consumed
/// directly; materialize such an operand into a contiguous tensor. A broadcast
/// *batch* dim is only `MildlyPermuted` and stays untouched (handled natively).
///
/// A storage-tiled operand is passed as it is: its strides order its tiles, not a matrix's rows
/// and columns, so this classification does not read them, and a routine that reads storage
/// tiles checks the layout it was handed itself.
#[allow(clippy::result_large_err)]
pub(crate) fn into_contiguous_if_highly_permuted(
    client: &Client,
    binding: InputBinding,
) -> Result<InputBinding, MatmulSetupError> {
    if binding.data().tiling.is_tiled() {
        return Ok(binding);
    }
    match matrix_batch_layout(&binding.data().strides, binding.scheme()) {
        MatrixBatchLayout::HighlyPermuted => Ok(binding.into_contiguous(client)?),
        _ => Ok(binding),
    }
}

/// The contract to solve a matmul
pub trait Routine<RC: RuntimeConfig>:
    cubek_std::Routine<Strategy: Default + Display + Clone, Blueprint: Debug + Clone> + Sized
{
}

/// What the routine reads about the device before it can shape a blueprint.
pub struct DeviceSettings {
    pub client: Client,
    pub plane_dim: u32,
    pub vector_sizes: MatmulVectorSizes,
    pub max_cube_count: (u32, u32, u32),
}

pub use cubek_std::BlueprintStrategy;
