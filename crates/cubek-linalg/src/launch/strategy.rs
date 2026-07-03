use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::definition::QRSetupError;
use crate::launch::QRTuple;

/// Define the strategy to use when calling for a QR decomposition.
///
/// Each variant maps to a routine in [`crate::routines`] that adapts the
/// corresponding algorithm to the problem and the hardware.
#[derive(Debug, Clone, Default)]
pub enum QRStrategy {
    /// Numerically stable, dense matrix using blocked Householder reflectors.
    BlockedAcceleratedHouseHolder,
    /// TSQR-inspired: entire panel in one fused kernel per tile (min dispatches).
    BahtTsqr,
    /// Performs the QR decomposition using Givens rotations.
    /// Better for sparse matrices and less numerically stable than Householder transformations.
    CommonGivensRotations,
    /// Modified Gram-Schmidt orthogonalization. Single persistent-cube kernel per
    /// column with no GEMM dispatch overhead — best for small matrices.
    ModifiedGramSchmidt,
    /// Automatically choose the best strategy for the problem, resolved by
    /// [`crate::launch::select_strategy`].
    #[default]
    Auto,
}

impl QRStrategy {
    /// It launches a QR decomposition over a m x n matrix a.
    ///
    /// Specify a strategy for the QR decomposition, the client and the matrix a to decompose.
    /// In case of success it will return a tuple with the matrix Q and the matrix R in this order.
    pub fn launch<R: Runtime, EG: Float + CubeElement>(
        &self,
        client: &ComputeClient<R>,
        a: &TensorHandle<R>,
    ) -> Result<QRTuple<R>, QRSetupError> {
        crate::launch::launch::<R, EG>(self, client, a)
    }

    /// Solves the system of equations Ax = b using the QR decomposition.
    pub fn solve<R: Runtime, EG: Float + CubeElement>(
        &self,
        client: &ComputeClient<R>,
        a: &TensorHandle<R>,
        b: &TensorHandle<R>,
    ) -> Result<TensorHandle<R>, QRSetupError> {
        crate::launch::solve::<R, EG>(self, client, a, b)
    }
}
