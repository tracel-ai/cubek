/// TODO move MatrixLayout to std, then use enum
pub fn compute_strides(shape: &[usize], col_major: bool) -> Vec<usize> {
    let n = shape.len();
    assert!(n >= 2, "Matrix must have at least 2 dimensions");

    let mut strides = Vec::with_capacity(n);
    let mut acc = 1;

    for &dim in shape.iter().rev() {
        strides.push(acc);
        acc *= dim;
    }
    strides.reverse();

    if col_major {
        strides.swap(n - 1, n - 2);
    }

    strides
}
