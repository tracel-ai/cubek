use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::{
    components::solve::{back_substitution_kernel, q_already_t_b_kernel},
    definition::QRSetupError,
};

/// Solve Ax = b using QR decomposition.
pub fn solve<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
    b: &TensorHandle<R>,
) -> Result<TensorHandle<R>, QRSetupError> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a.len() != 2 || shape_b.len() != 1 || shape_a[0] != shape_b[0] {
        return Err(QRSetupError::InvalidShape);
    }

    let m = shape_a[0];
    let n = shape_a[1];

    if m < n {
        return Err(QRSetupError::InvalidShape);
    }

    // 1. A = QR
    let (q, r) = crate::launch::qr::<R, E>(client, a)?;

    // 2. y = Q^T * b
    let y = TensorHandle::zeros(client, vec![m], a.dtype);
    let max_cube_dim = client.properties().hardware.max_cube_dim.0;
    let cd_q = CubeDim::new_1d(max_cube_dim.min(m as u32));
    let cc_q = calculate_cube_count_elemwise(client, m, cd_q);

    // All strategies return Q^T
    unsafe {
        q_already_t_b_kernel::launch_unchecked::<E, R>(
            client,
            cc_q,
            cd_q,
            m as u32,
            q.clone().into_arg(),
            b.clone().into_arg(),
            y.clone().into_arg(),
        );
    }

    // 3. Rx = y (first n elements of y if m > n)
    let x = TensorHandle::zeros(client, vec![n], a.dtype);
    let is_col_major = 1u32;

    // For back substitution, we use a single cube since it's sequential
    unsafe {
        back_substitution_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            n as u32,
            m as u32,
            r.clone().into_arg(),
            y.clone().into_arg(),
            x.clone().into_arg(),
            is_col_major,
        );
    }

    Ok(x)
}
