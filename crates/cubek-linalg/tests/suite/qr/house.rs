use std::fmt::Display;

use cubecl::{TestRuntime, prelude::*, std::tensor::{TensorHandle, into_contiguous}};

use crate::suite::utils::{
    assert_equals_approx, dtype_unsupported, tensorhandler_from_data,
    tensorhandler_from_data_col_major,
};

/// Run into_contiguous on a tensor and read the resulting tight row-major bytes.
fn read_contig<F: Float + CubeElement>(
    client: &ComputeClient<TestRuntime>,
    t: &TensorHandle<TestRuntime>,
) -> (Vec<F>, Vec<usize>) {
    let shape = t.shape().to_vec();
    let contig = into_contiguous::<TestRuntime>(client, t.clone().binding(), t.dtype);
    let bytes = client.read_one(contig.handle.clone()).unwrap();
    (F::from_bytes(&bytes).to_vec(), shape)
}

/// Reconstruct A = Q * R in f64 for maximum verification accuracy.
/// Q^T is row-major (after into_contiguous): Q[i,k] = q_t[k * rows + i].
/// R is row-major (after into_contiguous):   R[k,j] = r[k * cols + j].
fn reconstruct_qr<F: Float + CubeElement>(
    q_t_vals: &[F],
    r_vals: &[F],
    rows: usize,
    cols: usize,
    k_range: usize,
    q_row_stride: usize,
) -> Vec<F> {
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f64;
            for k in 0..k_range {
                let q_ik = q_t_vals[k * q_row_stride + i].to_f64().unwrap();
                let r_kj = r_vals[k * cols + j].to_f64().unwrap();
                sum += q_ik * r_kj;
            }
            out[i * cols + j] = sum;
        }
    }
    out.iter().map(|&v| <F as num_traits::NumCast>::from(v).unwrap()).collect()
}

pub fn test_qr_baht<F: Float + CubeElement + Display>(dim: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let dim_usize = dim as usize;

    let shape = vec![dim_usize, dim_usize];
    let num_elements = shape.iter().product();
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }

    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    let (q_t, r) = match cubek_linalg::launch::<TestRuntime, F>(
        &cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder,
        &client,
        &a,
    ) {
        Ok((q_t, r)) => (q_t, r),
        Err(_) => (
            TensorHandle::empty(&client, shape.clone(), a.dtype),
            TensorHandle::empty(&client, shape.clone(), a.dtype),
        ),
    };

    let (q_t_vals, _) = read_contig::<F>(&client, &q_t);
    let (r_vals_out, _) = read_contig::<F>(&client, &r);

    // Reconstruct in f64 for accurate verification.
    let out_data = reconstruct_qr(&q_t_vals, &r_vals_out, dim_usize, dim_usize, dim_usize, dim_usize);

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &data, 2e-3) {
        panic!("{}", e);
    }
}

pub fn test_qr_baht_rect<F: Float + CubeElement + Display>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let shape = vec![rows_usize, cols_usize];
    let num_elements = rows_usize * cols_usize;

    let mut row_major_data = vec![F::from_int(1); num_elements];
    for i in 0..rows_usize.min(cols_usize) {
        row_major_data[i * cols_usize + i] = F::from_int(2);
    }

    let mut col_major_data = vec![F::from_int(0); num_elements];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            col_major_data[j * rows_usize + i] = row_major_data[i * cols_usize + j];
        }
    }

    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &col_major_data,
        F::as_type_native_unchecked(),
    );

    let (q_t, r) = match cubek_linalg::launch::<TestRuntime, F>(
        &cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder,
        &client,
        &a,
    ) {
        Ok((q_t, r)) => (q_t, r),
        Err(e) => panic!("QR launch failed: {:?}", e),
    };

    // Q^T row-major [rows × rows], R row-major [rows × cols].
    // q_t_vals[k * rows + i] = Q^T[k, i] = Q[i, k].
    let (q_t_vals, qt_shape) = read_contig::<F>(&client, &q_t);
    let (r_vals_out, _) = read_contig::<F>(&client, &r);

    let out_data = reconstruct_qr(&q_t_vals, &r_vals_out, rows_usize, cols_usize, qt_shape[0], qt_shape[1]);

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &row_major_data, 2e-3) {
        panic!("{}", e);
    }
}

pub fn test_qr_tsqr<F: Float + CubeElement + Display>(dim: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let dim_usize = dim as usize;

    let shape = vec![dim_usize, dim_usize];
    let num_elements = shape.iter().product();
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }

    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    let (q_t, r) = match cubek_linalg::launch::<TestRuntime, F>(
        &cubek_linalg::QRStrategy::BahtTsqr,
        &client,
        &a,
    ) {
        Ok((q_t, r)) => (q_t, r),
        Err(_) => (
            TensorHandle::empty(&client, shape.clone(), a.dtype),
            TensorHandle::empty(&client, shape.clone(), a.dtype),
        ),
    };

    let (q_t_vals, _) = read_contig::<F>(&client, &q_t);
    let (r_vals_out, _) = read_contig::<F>(&client, &r);

    let out_data = reconstruct_qr(&q_t_vals, &r_vals_out, dim_usize, dim_usize, dim_usize, dim_usize);

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &data, 2e-3) {
        panic!("{}", e);
    }
}

pub fn test_qr_tsqr_rect<F: Float + CubeElement + Display>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let shape = vec![rows_usize, cols_usize];
    let num_elements = rows_usize * cols_usize;

    let mut row_major_data = vec![F::from_int(1); num_elements];
    for i in 0..rows_usize.min(cols_usize) {
        row_major_data[i * cols_usize + i] = F::from_int(2);
    }

    let mut col_major_data = vec![F::from_int(0); num_elements];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            col_major_data[j * rows_usize + i] = row_major_data[i * cols_usize + j];
        }
    }

    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &col_major_data,
        F::as_type_native_unchecked(),
    );

    let (q_t, r) = match cubek_linalg::launch::<TestRuntime, F>(
        &cubek_linalg::QRStrategy::BahtTsqr,
        &client,
        &a,
    ) {
        Ok((q_t, r)) => (q_t, r),
        Err(e) => panic!("QR launch failed: {:?}", e),
    };

    let (q_t_vals, qt_shape) = read_contig::<F>(&client, &q_t);
    let (r_vals_out, _) = read_contig::<F>(&client, &r);

    let out_data = reconstruct_qr(&q_t_vals, &r_vals_out, rows_usize, cols_usize, qt_shape[0], qt_shape[1]);

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &row_major_data, 2e-3) {
        panic!("{}", e);
    }
}
