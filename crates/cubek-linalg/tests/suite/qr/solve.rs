use crate::suite::utils::{
    assert_equals_approx, dtype_unsupported, tensorhandler_from_data,
    tensorhandler_from_data_col_major,
};
use cubecl::{TestRuntime, prelude::*};
use std::fmt::Display;

pub fn test_solve_square<F: Float + CubeElement + Display>(dim: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let dim_usize = dim as usize;

    let is_col_major = true;

    // Create a diagonally dominant matrix A
    let mut a_data = vec![F::from_int(0); dim_usize * dim_usize];
    for i in 0..dim_usize {
        for j in 0..dim_usize {
            let val = if i == j {
                F::from_int(10)
            } else {
                F::from_int(1)
            };
            if is_col_major {
                a_data[j * dim_usize + i] = val;
            } else {
                a_data[i * dim_usize + j] = val;
            }
        }
    }

    let a = if is_col_major {
        tensorhandler_from_data_col_major(
            &client,
            vec![dim_usize, dim_usize],
            &a_data,
            F::as_type_native_unchecked(),
        )
    } else {
        tensorhandler_from_data(
            &client,
            vec![dim_usize, dim_usize],
            &a_data,
            F::as_type_native_unchecked(),
        )
    };

    // Create a true solution x_true
    let x_true_data: Vec<F> = (0..dim_usize)
        .map(|i| F::from_int((i + 1) as i64))
        .collect();

    // Compute b = A * x_true
    let mut b_data = vec![F::from_int(0); dim_usize];
    for i in 0..dim_usize {
        let mut sum = F::from_int(0);
        for j in 0..dim_usize {
            let a_val = if is_col_major {
                a_data[j * dim_usize + i]
            } else {
                a_data[i * dim_usize + j]
            };
            sum += a_val * x_true_data[j];
        }
        b_data[i] = sum;
    }

    let b = tensorhandler_from_data(
        &client,
        vec![dim_usize],
        &b_data,
        F::as_type_native_unchecked(),
    );

    let x = cubek_linalg::solve::<TestRuntime, F>(&client, &a, &b).unwrap();

    assert_equals_approx(&client, &x, &x_true_data, 5e-2).unwrap();
}

pub fn test_solve_rect<F: Float + CubeElement + Display>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    if dtype_unsupported::<F>(&client) {
        return;
    }
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let is_col_major = true;

    // Create a tall matrix A
    let mut a_data = vec![F::from_int(0); rows_usize * cols_usize];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            let val = if i == j {
                F::from_int(10)
            } else {
                F::from_int(1)
            };
            if is_col_major {
                a_data[j * rows_usize + i] = val;
            } else {
                a_data[i * cols_usize + j] = val;
            }
        }
    }

    let a = if is_col_major {
        tensorhandler_from_data_col_major(
            &client,
            vec![rows_usize, cols_usize],
            &a_data,
            F::as_type_native_unchecked(),
        )
    } else {
        tensorhandler_from_data(
            &client,
            vec![rows_usize, cols_usize],
            &a_data,
            F::as_type_native_unchecked(),
        )
    };

    // Create a true solution x_true
    let x_true_data: Vec<F> = (0..cols_usize)
        .map(|i| F::from_int((i + 1) as i64))
        .collect();

    // Compute b = A * x_true
    let mut b_data = vec![F::from_int(0); rows_usize];
    for i in 0..rows_usize {
        let mut sum = F::from_int(0);
        for j in 0..cols_usize {
            let a_val = if is_col_major {
                a_data[j * rows_usize + i]
            } else {
                a_data[i * cols_usize + j]
            };
            sum += a_val * x_true_data[j];
        }
        b_data[i] = sum;
    }

    let b = tensorhandler_from_data(
        &client,
        vec![rows_usize],
        &b_data,
        F::as_type_native_unchecked(),
    );

    let x = cubek_linalg::solve::<TestRuntime, F>(&client, &a, &b).unwrap();

    assert_equals_approx(&client, &x, &x_true_data, 5e-2).unwrap();
}
