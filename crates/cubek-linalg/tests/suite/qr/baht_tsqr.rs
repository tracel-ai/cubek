use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{TensorHandle, into_contiguous},
};

use crate::suite::utils::{
    assert_equals_approx, col_major_input, dtype_unsupported, row_major_input,
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
    out.iter()
        .map(|&v| <F as num_traits::NumCast>::from(v).unwrap())
        .collect()
}

fn run_qr_square<F: Float + CubeElement>(dim: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let dim_usize = dim as usize;

    let shape = vec![dim_usize, dim_usize];
    let num_elements = shape.iter().product();
    // Ones with 2s on the anti-diagonal (a symmetric matrix, logical
    // row-major).
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }

    let a = col_major_input(&client, shape.clone(), &data);

    let (q_t, r) = match cubek_linalg::qr::<TestRuntime, F>(&client, &a) {
        Ok((q_t, r)) => (q_t, r),
        Err(e) => panic!("QR launch failed: {e:?}"),
    };

    let (q_t_vals, _) = read_contig::<F>(&client, &q_t);
    let (r_vals_out, _) = read_contig::<F>(&client, &r);

    // Reconstruct in f64 for accurate verification.
    let out_data = reconstruct_qr(
        &q_t_vals,
        &r_vals_out,
        dim_usize,
        dim_usize,
        dim_usize,
        dim_usize,
    );

    assert_equals_approx::<F>(&out_data, &data, shape, 2e-3);
}

fn run_qr_rect<F: Float + CubeElement>(rows: u32, cols: u32, row_major: bool) {
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

    // `qr` normalizes any input layout to its internal col-major form, so
    // both layouts must factorize the same logical matrix identically.
    let a = if row_major {
        row_major_input(&client, shape.clone(), &row_major_data)
    } else {
        col_major_input(&client, shape.clone(), &row_major_data)
    };

    let (q_t, r) = match cubek_linalg::qr::<TestRuntime, F>(&client, &a) {
        Ok((q_t, r)) => (q_t, r),
        Err(e) => panic!("QR launch failed: {e:?}"),
    };

    // Q^T row-major [rows × rows], R row-major [rows × cols].
    // q_t_vals[k * rows + i] = Q^T[k, i] = Q[i, k].
    let (q_t_vals, qt_shape) = read_contig::<F>(&client, &q_t);
    let (r_vals_out, _) = read_contig::<F>(&client, &r);

    let out_data = reconstruct_qr(
        &q_t_vals,
        &r_vals_out,
        rows_usize,
        cols_usize,
        qt_shape[0],
        qt_shape[1],
    );

    assert_equals_approx::<F>(&out_data, &row_major_data, shape, 2e-3);
}

pub fn test_qr<F: Float + CubeElement>(dim: u32) {
    run_qr_square::<F>(dim);
}

pub fn test_qr_rect<F: Float + CubeElement>(rows: u32, cols: u32) {
    run_qr_rect::<F>(rows, cols, false);
}

pub fn test_qr_rect_row_major<F: Float + CubeElement>(rows: u32, cols: u32) {
    run_qr_rect::<F>(rows, cols, true);
}
