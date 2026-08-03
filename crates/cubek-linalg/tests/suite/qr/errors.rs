//! Validation tests for the launch-layer guards: shapes and element types
//! must be rejected with a `QRSetupError` before any kernel is dispatched.

use cubecl::{TestRuntime, prelude::*};
use cubek_linalg::QRSetupError;

use crate::suite::utils::col_major_input;

#[test]
fn qr_rejects_zero_columns() {
    let client = TestRuntime::client(&Default::default());
    let a = col_major_input::<f32>(&client, vec![5, 0], &[]);
    let result = cubek_linalg::qr::<TestRuntime, f32>(&client, &a);
    assert_eq!(result.err(), Some(QRSetupError::InvalidShape));
}

#[test]
fn qr_rejects_wide_matrix() {
    let client = TestRuntime::client(&Default::default());
    let a = col_major_input::<f32>(&client, vec![2, 4], &[1.0f32; 8]);
    let result = cubek_linalg::qr::<TestRuntime, f32>(&client, &a);
    assert_eq!(result.err(), Some(QRSetupError::InvalidShape));
}

#[test]
fn qr_rejects_dtype_mismatch() {
    let client = TestRuntime::client(&Default::default());
    // f64 tensor launched through the f32 entry point: must error out
    // before any buffer is sized or kernel dispatched.
    let a = col_major_input::<f64>(&client, vec![4, 2], &[1.0f64; 8]);
    let result = cubek_linalg::qr::<TestRuntime, f32>(&client, &a);
    assert!(matches!(result, Err(QRSetupError::TypeMismatch { .. })));
}

#[test]
fn solve_rejects_dtype_mismatch_b() {
    let client = TestRuntime::client(&Default::default());
    let a = col_major_input::<f32>(&client, vec![2, 2], &[2.0, 0.0, 0.0, 2.0f32]);
    let b = col_major_input::<f64>(&client, vec![2], &[1.0f64; 2]);
    let result = cubek_linalg::solve::<TestRuntime, f32>(&client, &a, &b);
    assert!(matches!(result, Err(QRSetupError::TypeMismatch { .. })));
}
