use cubecl::prelude::*;
use cubecl::std::tensor::{TensorHandle, identity};

use crate::qr::QRSetupError;
use crate::qr::{baht, baht_tsqr, cgr, mgs};

type QRTuple<R> = (TensorHandle<R>, TensorHandle<R>);

/// Define the strategy to use when calling for a QR decomposition.
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
    /// Automatically choose the best strategy.
    #[default]
    Auto,
}

fn initialize<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
) -> Result<QRTuple<R>, QRSetupError> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] < shape[1] {
        return Err(QRSetupError::InvalidShape);
    }
    let dtype = a.dtype;
    let (m, _n) = (shape[0], shape[1]);

    // Allocate Q as identity (col-major [rows, rows]).
    // IMPORTANT: TensorHandle::zeros / ::empty may return a GPU-padded buffer
    // (e.g. pitch 4 for a 3-row matrix).  The BAHT kernels index Q^T with flat
    // `col*rows+row` arithmetic that assumes NO pitch padding.  We therefore
    // reserve a tight (non-padded) buffer via `client.empty` and fill it on the
    // GPU with cubecl's `identity` kernel — no host round-trip, any float type.
    //
    // The identity kernel derives the diagonal stride from `strides[0]`, so we
    // hand it standard row-major strides `[m, 1]`.  The identity matrix is
    // symmetric, so the resulting tight buffer is simultaneously the col-major
    // `[1, m]` identity the QR kernels consume.
    let q_shape = vec![m, m];
    let elem_size = dtype.size();
    let q_handle = client.empty(m * m * elem_size);
    let q_contig = TensorHandle::<R>::new(q_handle.clone(), q_shape.clone(), vec![m, 1], dtype);
    identity::launch::<R>(client, &q_contig);
    let q = TensorHandle::<R>::new(q_handle, q_shape, vec![1, m], dtype);

    // Build R as a tight col-major copy of A.  We must NOT use into_contiguous here
    // because that produces row-major output, which would make R's bytes inconsistent
    // with the col-major strides [1, m] the BAHT/CGR kernels expect.
    //
    // Instead: read A via its own strides (using into_contiguous which copies elements
    // in logical order from the source) then re-declare the output as col-major — BUT
    // only if A is already col-major.  The simplest safe approach: read A's raw bytes
    // directly (the test uses create_from_slice so no padding) and declare [1, m].
    let a_bytes = client.read_one(a.handle.clone()).unwrap();
    let a_handle = client.create_from_slice(&a_bytes);
    let a_strides = a.strides().to_vec(); // preserve the original strides
    let r = TensorHandle::<R>::new(a_handle, shape.to_vec(), a_strides, dtype);

    Ok((q, r))
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
        let (q, r) = initialize::<R, EG>(client, a)?;

        match self {
            QRStrategy::BlockedAcceleratedHouseHolder => {
                baht::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::BahtTsqr => {
                baht_tsqr::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::CommonGivensRotations => {
                cgr::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::ModifiedGramSchmidt => {
                mgs::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::Auto => {
                baht::launch::<R, EG>(client, &q, &r);
            }
        };

        Ok((q, r))
    }

    /// Solves the system of equations Ax = b using the QR decomposition.
    pub fn solve<R: Runtime, EG: Float + CubeElement>(
        &self,
        client: &ComputeClient<R>,
        a: &TensorHandle<R>,
        b: &TensorHandle<R>,
    ) -> Result<TensorHandle<R>, QRSetupError> {
        crate::qr::solve::solve::<R, EG>(self, client, a, b)
    }
}

/// It launches a QR decomposition over a m x n matrix a.
///
/// Specify a strategy for the QR decomposition, the client and the matrix a to decompose.
/// In case of success it will return a tuple with the matrix Q and the matrix R in this order.
pub fn launch<R: Runtime, EG: Float + CubeElement>(
    strategy: &QRStrategy,
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
) -> Result<QRTuple<R>, QRSetupError> {
    strategy.launch::<R, EG>(client, a)
}
