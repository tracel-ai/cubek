//! Kernels used to solve `Ax = b` from a QR decomposition:
//! `y = Q^T·b` followed by back substitution on `Rx = y`.

use cubecl::prelude::*;

/// Kernel to compute y = Q_already_t * b
/// Q_already_t is m x m stored col-major as produced by the QR kernels
/// (`Q^T[i, j]` at flat index `j*m + i`). b is m x 1.
/// y_i = sum_j (Q_already_t)_ij * b_j
#[cube(launch_unchecked)]
pub fn q_already_t_b_kernel<F: Float>(m: u32, q_t: &Tensor<F>, b: &Tensor<F>, y: &mut Tensor<F>) {
    let i = ABSOLUTE_POS_X;
    if i < m {
        let mut sum = 0.0f64;
        for j in 0..m {
            sum = fma(
                f64::cast_from(q_t[j as usize * m as usize + i as usize]),
                f64::cast_from(b[j as usize]),
                sum,
            );
        }
        y[i as usize] = F::cast_from(sum);
    }
}

/// Back substitution for Rx = y where R is upper triangular, stored tight
/// col-major `[rows, n]` as produced by the QR kernels.
/// Accumulates in f64 to minimise rounding in the triangular solve.
#[cube(launch_unchecked)]
pub fn back_substitution_kernel<F: Float>(
    n: u32,
    rows: u32,
    r: &Tensor<F>,
    y: &Tensor<F>,
    x: &mut Tensor<F>,
) {
    if ABSOLUTE_POS == 0 {
        let mut i = n;
        while i > 0 {
            i -= 1;
            let mut sum = 0.0f64;
            for j in (i + 1)..n {
                sum = fma(
                    f64::cast_from(r[j as usize * rows as usize + i as usize]),
                    f64::cast_from(x[j as usize]),
                    sum,
                );
            }
            let diag_idx = i as usize * rows as usize + i as usize;
            x[i as usize] =
                F::cast_from((f64::cast_from(y[i as usize]) - sum) / f64::cast_from(r[diag_idx]));
        }
    }
}
