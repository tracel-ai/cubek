use cubecl::prelude::*;
use cubecl::std::tensor::{TensorHandle, identity, into_contiguous};

use crate::{
    components,
    definition::{QRProblem, QRSetupError},
    routines::{BahtTsqrRoutine, BahtTsqrStrategy, BlueprintStrategy, QRRoutine},
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
    let n = problem.cols;
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

    // Build R as a tight col-major copy of A, entirely on-device and accepting
    // any input layout (row-major, col-major, pitch-padded): a tight col-major
    // `[m, n]` buffer is byte-identical to a tight row-major `[n, m]` buffer of
    // the transpose, so run `into_contiguous` (tight `client.empty` output,
    // reads the source through its strides) on a transposed *view* of A and
    // re-declare the result with col-major strides `[1, m]`.
    let a_strides = a.strides();
    let a_t = TensorHandle::<R>::new(
        a.handle.clone(),
        vec![n, m],
        vec![a_strides[1], a_strides[0]],
        dtype,
    );
    let r_t = into_contiguous::<R>(client, a_t.binding(), dtype);
    let r = TensorHandle::<R>::new(r_t.handle, vec![m, n], vec![1, m], dtype);

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
    qr_with_strategy::<R, EG>(client, a, Default::default())
}

/// [`qr`] with explicit routine knobs.
///
/// The only knob today is [`BahtTsqrStrategy::allow_tf32`], which lets the
/// trailing-update GEMMs run on tensor cores: about 2x faster end to end, at a
/// reconstruction error near `1e-2` instead of f32 accuracy. [`qr`] leaves it
/// off, so its accuracy is unchanged.
pub fn qr_with_strategy<R: Runtime, EG: Float + CubeElement>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
    strategy: BahtTsqrStrategy,
) -> Result<QRTuple<R>, QRSetupError> {
    let problem = QRProblem::from_shape(a.shape(), a.dtype)?;
    let launched = EG::as_type_native_unchecked().storage_type();
    if problem.dtype != launched {
        return Err(QRSetupError::TypeMismatch {
            launched,
            actual: problem.dtype,
        });
    }
    let (q, r) = initialize::<R>(client, a, &problem);

    let (blueprint, settings) =
        BahtTsqrRoutine::prepare(client, &problem, BlueprintStrategy::Inferred(strategy))?;
    components::baht_tsqr::launch::<R, EG>(client, &q, &r, blueprint, settings)?;

    Ok((q, r))
}
