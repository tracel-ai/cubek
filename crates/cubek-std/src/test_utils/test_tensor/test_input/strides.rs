#[derive(Debug, PartialEq, Eq, Default)]
pub enum StrideSpec {
    #[default]
    RowMajor,
    ColMajor,
    Custom(Vec<usize>),
}

impl StrideSpec {
    pub fn compute_strides(&self, shape: &[usize]) -> Vec<usize> {
        let n = shape.len();

        match self {
            StrideSpec::RowMajor => {
                assert!(n >= 2, "RowMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 1] = 1;
                for i in (0..n - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                strides
            }
            StrideSpec::ColMajor => {
                assert!(n >= 2, "ColMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 2] = 1;
                strides[n - 1] = shape[n - 2];
                for i in (0..n - 2).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                strides
            }
            StrideSpec::Custom(strides) => {
                assert!(
                    strides.len() == n,
                    "Custom strides must have the same rank as the shape"
                );
                strides.clone()
            }
        }
    }
}

/// Reorders a flat array according to given strides.
pub(crate) fn reorder_by_strides<T: Copy + Default>(
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
