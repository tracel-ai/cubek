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
#[allow(clippy::result_large_err)]
pub(crate) fn into_contiguous_if_highly_permuted<R: Runtime>(
    client: &ComputeClient<R>,
    binding: InputBinding<R>,
) -> Result<InputBinding<R>, MatmulSetupError> {
    match matrix_batch_layout(&binding.data().strides, binding.scheme()) {
        MatrixBatchLayout::HighlyPermuted => Ok(binding.into_contiguous(client)?),
        _ => Ok(binding),
    }
}

/// The contract to solve a matmul
pub trait Routine<RC: RuntimeConfig>: Sized {
    type Strategy: Default + Display + Clone;
    type Blueprint: Debug + Clone;
}

/// What the routine reads about the device before it can shape a blueprint.
pub struct DeviceSettings<R: Runtime> {
    pub client: ComputeClient<R>,
    pub plane_dim: u32,
    pub vector_sizes: MatmulVectorSizes,
    pub max_cube_count: (u32, u32, u32),
}

pub enum BlueprintStrategy<RC: RuntimeConfig, A: Routine<RC>> {
    /// Use a predefined blueprint
    Forced(A::Blueprint),
    /// Allows to give limited blueprint information, and the rest is inferred from it
    Inferred(A::Strategy),
}

impl<RC: RuntimeConfig, A: Routine<RC>> BlueprintStrategy<RC, A> {
    pub fn maybe_forced_default(s: &Option<A::Blueprint>) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_default()
    }
    pub fn maybe_forced_or(s: &Option<A::Blueprint>, args: &A::Strategy) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_else(|| Self::Inferred(args.clone()))
    }
}

impl<RC: RuntimeConfig, A: Routine<RC>> Default for BlueprintStrategy<RC, A> {
    fn default() -> Self {
        Self::Inferred(Default::default())
    }
}

impl<RC: RuntimeConfig, A: Routine<RC>> Display for BlueprintStrategy<RC, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forced(_) => f.write_str("_forced"),
            Self::Inferred(strategy) => write!(f, "{}", strategy),
        }
    }
}

impl<RC: RuntimeConfig, A: Routine<RC>> Clone for BlueprintStrategy<RC, A> {
    fn clone(&self) -> Self {
        match self {
            Self::Forced(blueprint) => Self::Forced(blueprint.clone()),
            Self::Inferred(strategy) => Self::Inferred(strategy.clone()),
        }
    }
}
