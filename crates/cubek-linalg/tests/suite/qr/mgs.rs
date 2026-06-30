use std::fmt::Display;

use cubecl::{TestRuntime, prelude::*, std::tensor::{TensorHandle, into_contiguous}};

use crate::suite::utils::{
    assert_equals_approx, dtype_unsupported, tensorhandler_from_data,
    tensorhandler_from_data_col_major, transpose_matrix,
};

pub fn test_qr_mgs<F: Float + CubeElement + Display>(dim: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let dim_usize = dim as usize;

    let shape = vec![dim as usize, dim as usize];
    let num_elements = shape.iter().product();
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }

    // MGS (like BAHT/CGR) processes R internally as col-major; use col-major input.
    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    let (mut q_t, r) =
        match cubek_linalg::launch::<TestRuntime, F>(&cubek_linalg::QRStrategy::ModifiedGramSchmidt, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(_) => (
                TensorHandle::empty(&client, shape.clone(), a.dtype),
                TensorHandle::empty(&client, shape.clone(), a.dtype),
            ),
        };

    // transpose_matrix already calls into_contiguous internally, giving row-major Q.
    // Q[i,k] = q_vals[i * dim + k]  ✓
    let q = transpose_matrix(&client, &mut q_t);
    let bytes_q = client.read_one(q.handle.clone()).unwrap();
    let q_vals = F::from_bytes(&bytes_q);

    // into_contiguous on col-major R → row-major: R[k,j] = r_vals[k * dim + j]
    let r_contig = into_contiguous::<TestRuntime>(&client, r.clone().binding(), r.dtype);
    let bytes_r = client.read_one(r_contig.handle.clone()).unwrap();
    let r_vals = F::from_bytes(&bytes_r);

    let mut out_data = vec![F::from_int(0); num_elements];
    let dim_usize = dim as usize;
    for i in 0..dim_usize {
        for j in 0..dim_usize {
            let mut sum = 0.0f64;
            for k in 0..dim_usize {
                sum += q_vals[i * dim_usize + k].to_f64().unwrap()
                    * r_vals[k * dim_usize + j].to_f64().unwrap();
            }
            out_data[i * dim_usize + j] = <F as num_traits::NumCast>::from(sum).unwrap();
        }
    }

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &data, 2e-3) {
        panic!("{}", e);
    }
}

pub fn test_qr_mgs_rect<F: Float + CubeElement + Display>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let shape = vec![rows_usize, cols_usize];
    let num_elements = rows_usize * cols_usize;

    // Build row-major data then convert to col-major for the algorithm.
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

    let (mut q_t, r) =
        match cubek_linalg::launch::<TestRuntime, F>(&cubek_linalg::QRStrategy::ModifiedGramSchmidt, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(e) => panic!("QR launch failed: {:?}", e),
        };

    let q = transpose_matrix(&client, &mut q_t);
    let bytes_q = client.read_one(q.handle.clone()).unwrap();
    let q_vals = F::from_bytes(&bytes_q);
    let q_shape = q.shape();

    let r_contig = into_contiguous::<TestRuntime>(&client, r.clone().binding(), r.dtype);
    let bytes_r = client.read_one(r_contig.handle.clone()).unwrap();
    let r_vals = F::from_bytes(&bytes_r);
    let r_shape = r.shape();

    let mut out_data = vec![F::from_int(0); num_elements];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            let mut sum = 0.0f64;
            for k in 0..q_shape[1] {
                // Q row-major: Q[i,k] = q_vals[i * q_cols + k]
                // R row-major (after contig): R[k,j] = r_vals[k * r_cols + j]
                sum += q_vals[i * q_shape[1] + k].to_f64().unwrap()
                    * r_vals[k * r_shape[1] + j].to_f64().unwrap();
            }
            out_data[i * cols_usize + j] = <F as num_traits::NumCast>::from(sum).unwrap();
        }
    }

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &row_major_data, 2e-3) {
        panic!("{}", e);
    }
}
