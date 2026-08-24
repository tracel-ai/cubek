use cubecl::prelude::*;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubek_std::InputBinding;
use cubek_tile::{Axis, Operand, OperandSet};
use std::fmt::{Debug, Display};

use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes};

/// A launch-time config carried alongside the tensors. `()` for a plain matmul; the
/// fusion path substitutes its own.
pub trait RuntimeConfig: LaunchArg + CubeType<ExpandType: Clone> + Clone + Send + Sync {}
impl<T: LaunchArg + CubeType<ExpandType: Clone> + Clone + Send + Sync> RuntimeConfig for T {}

// The matmul tile axes, shared by every routine that lays out its space over `(m, n, k)` plus
// batches. `M`/`N`/`K` are the two matrix dims and the contraction; batch axes follow.
pub(crate) const M: Axis = Axis(0);
pub(crate) const N: Axis = Axis(1);
pub(crate) const K: Axis = Axis(2);

/// The axis for output batch dimension `i` (outermost is `0`).
pub(crate) fn batch_axis(i: usize) -> Axis {
    Axis(3 + i as u8)
}

/// The three matmul operands as [`Operand`]s for a [`Tiling::over`](cubek_tile::Tiling::over)
/// build: `a` over `[M, K]`, `b` over `[K, N]`, `out` over `[M, N]`, each at its global
/// element type. Batch axes are per-binding broadcast facts, stated at launch, not here.
pub(crate) struct MatmulOperands {
    pub a: Operand,
    pub b: Operand,
    pub out: Operand,
}

impl MatmulOperands {
    pub fn new(dtypes: &MatmulElems) -> Self {
        MatmulOperands {
            a: Operand::new(&[M, K], dtypes.lhs_global),
            b: Operand::new(&[K, N], dtypes.rhs_global),
            out: Operand::new(&[M, N], dtypes.acc_global),
        }
    }
}

impl OperandSet for MatmulOperands {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.a, &mut self.b, &mut self.out].into_iter()
    }
}

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
