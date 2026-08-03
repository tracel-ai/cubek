use crate::suite::utils::{assert_equals_approx, col_major_input, dtype_unsupported};
use cubecl::{TestRuntime, prelude::*};

/// Build a diagonally dominant `[rows, cols]` matrix (logical row-major),
/// a known solution `x_true`, and the matching right-hand side `b = A·x_true`.
fn make_system<F: Float + CubeElement>(rows: usize, cols: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut a_data = vec![F::from_int(0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            a_data[i * cols + j] = if i == j {
                F::from_int(10)
            } else {
                F::from_int(1)
            };
        }
    }

    let x_true: Vec<F> = (0..cols).map(|i| F::from_int((i + 1) as i64)).collect();

    let mut b_data = vec![F::from_int(0); rows];
    for i in 0..rows {
        let mut sum = F::from_int(0);
        for j in 0..cols {
            sum += a_data[i * cols + j] * x_true[j];
        }
        b_data[i] = sum;
    }

    (a_data, x_true, b_data)
}

fn run_solve<F: Float + CubeElement>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let (a_data, x_true, b_data) = make_system::<F>(rows_usize, cols_usize);

    let a = col_major_input(&client, vec![rows_usize, cols_usize], &a_data);
    let b = col_major_input(&client, vec![rows_usize], &b_data);

    let x = cubek_linalg::solve::<TestRuntime, F>(&client, &a, &b).unwrap();

    let x_bytes = client.read_one(x.handle.clone()).unwrap();
    let x_vals = F::from_bytes(&x_bytes).to_vec();

    assert_equals_approx::<F>(&x_vals, &x_true, vec![cols_usize], 5e-2);
}

pub fn test_solve_square<F: Float + CubeElement>(dim: u32) {
    run_solve::<F>(dim, dim);
}

pub fn test_solve_rect<F: Float + CubeElement>(rows: u32, cols: u32) {
    run_solve::<F>(rows, cols);
}
