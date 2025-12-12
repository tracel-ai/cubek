/// TODO move MatrixLayout to std, then use enum
pub fn contiguous_strides(shape: &[usize], col_major: bool) -> Vec<usize> {
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

/// Reorders a flat array according to given strides.
///
/// # Example
/// ```
/// let flat = vec![1, 2, 3, 4];
/// let shape = vec![2, 2];
/// let strides = vec![1, 2];
/// let reordered = reorder_by_strides(&flat, &shape, &strides);
/// assert_eq!(reordered, vec![1, 3, 2, 4]);
/// ```
pub fn reorder_by_strides<T: Copy + Default>(
    flat: &[T],
    shape: &[usize],
    strides: &[usize],
) -> Vec<T> {
    let total = flat.len();
    let mut out = vec![T::default(); total];

    let rank = shape.len();
    let mut index = vec![0usize; rank];

    #[allow(clippy::needless_range_loop)]
    for logical_flat_idx in 0..total {
        // Compute multi-dim index in row-major order
        let mut remaining = logical_flat_idx;
        for d in (0..rank).rev() {
            let dim = shape[d];
            index[d] = remaining % dim;
            remaining /= dim;
        }

        // Compute physical offset using custom strides
        let mut physical = 0usize;
        for d in 0..rank {
            physical += index[d] * strides[d];
        }

        out[logical_flat_idx] = flat[physical];
    }

    out
}
