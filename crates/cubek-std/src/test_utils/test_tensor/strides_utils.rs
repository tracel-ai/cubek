/// Compute strides for a batched matrix tensor.
///
/// Last two dims are treated as a matrix; preceding dims are batches.
/// By default row-major. Set `col_major` to true to swap the last two strides.
pub fn batched_matrix_strides(shape: &[usize], col_major: bool) -> Vec<usize> {
    let n = shape.len();
    assert!(n >= 2);

    let mut strides = vec![0; n];

    if col_major {
        // column-major for last two dims
        strides[0] = 1;
        for i in 1..n {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
    } else {
        // row-major
        strides[n - 1] = 1;
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    strides
}

// /// Reorders a flat array according to given strides.
// pub fn reorder_by_strides<T: Copy + Default>(
//     flat: &[T],
//     shape: &[usize],
//     strides: &[usize],
// ) -> Vec<T> {
//     let total = flat.len();
//     let mut out = vec![T::default(); total];

//     let rank = shape.len();
//     let mut index = vec![0usize; rank];

//     #[allow(clippy::needless_range_loop)]
//     for logical_flat_idx in 0..total {
//         // Compute multi-dim index in row-major order
//         let mut remaining = logical_flat_idx;
//         for d in (0..rank).rev() {
//             let dim = shape[d];
//             index[d] = remaining % dim;
//             remaining /= dim;
//         }

//         // Compute physical offset using custom strides
//         let mut physical = 0usize;
//         for d in 0..rank {
//             physical += index[d] * strides[d];
//         }

//         out[logical_flat_idx] = flat[physical];
//     }

//     out
// }
