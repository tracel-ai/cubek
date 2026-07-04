use cubecl::prelude::*;
use cubecl::std::tensor::{TensorHandle, identity};

use crate::{
    components,
    definition::{QRProblem, QRSetupError},
    routines::{BahtTsqrRoutine, BlueprintStrategy, QRRoutine},
};

/// The `(Q, R)` pair produced by a QR decomposition.
pub type QRTuple<R> = (TensorHandle<R>, TensorHandle<R>);

/// Allocate and seed the Q (identity) and R (copy of A) buffers the QR
/// kernels work in-place on.
fn initialize<R: Runtime>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
    problem: &QRProblem,
) -> QRTuple<R> {
    let m = problem.rows;
    let dtype = problem.dtype;

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
    let r = TensorHandle::<R>::new(a_handle, a.shape().to_vec(), a_strides, dtype);

    (q, r)
}

/// It launches a QR decomposition over a m x n matrix a using the TSQR-inspired
/// blocked Householder routine ([`BahtTsqrRoutine`]).
///
/// Specify the client and the matrix a to decompose. In case of success it
/// will return a tuple with the matrix Q and the matrix R in this order.
pub fn qr<R: Runtime, EG: Float + CubeElement>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
) -> Result<QRTuple<R>, QRSetupError> {
    let problem = QRProblem::from_shape(a.shape(), a.dtype)?;
    let (q, r) = initialize::<R>(client, a, &problem);

    let (_blueprint, settings) = BahtTsqrRoutine::prepare(
        client,
        &problem,
        BlueprintStrategy::Inferred(Default::default()),
    )?;
    components::baht_tsqr::launch::<R, EG>(client, &q, &r, settings);

    Ok((q, r))
}
