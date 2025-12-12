/// Compute strides for a batched matrix tensor.
///
/// Last two dims are treated as a matrix; preceding dims are batches.
/// By default row-major. Set `col_major` to true to swap the last two strides.
pub fn batched_matrix_strides(shape: &[usize], col_major: bool) -> Vec<usize> {
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
