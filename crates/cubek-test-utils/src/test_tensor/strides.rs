use cubecl::zspace::{Shape, Strides};

/// How the base strides of a tensor are laid out before any tile-level rank
/// expansion is applied.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum StridedLayout {
    #[default]
    RowMajor,
    ColMajor,
    Explicit(Vec<usize>),
}

/// The layout of a test tensor: a base [`StridedLayout`] that drives the
/// physical strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutSpec {
    Strided(StridedLayout),
}

impl Default for LayoutSpec {
    fn default() -> Self {
        Self::Strided(StridedLayout::default())
    }
}

impl LayoutSpec {
    /// The base stride layout.
    pub fn base(&self) -> &StridedLayout {
        match self {
            LayoutSpec::Strided(base) => base,
        }
    }

    /// Compute the strides of the layout for `shape`.
    pub fn compute_strides(&self, shape: &Shape) -> Strides {
        self.base().compute_strides(shape)
    }
}

impl From<StridedLayout> for LayoutSpec {
    fn from(base: StridedLayout) -> Self {
        Self::Strided(base)
    }
}

/// Number of elements in the physical buffer required to cover every logical
/// index in `shape` under `strides`, assuming element 0 is at offset 0.
///
/// Exceeds `shape.iter().product()` for jumpy strides (e.g. a slice stepping
/// over padding) and is less than it for broadcast strides (a stride of 0
/// makes every index in that dim share the same physical offset).
pub fn physical_extent(shape: &Shape, strides: &Strides) -> usize {
    let mut max_offset = 0usize;
    for (s, d) in strides.iter().zip(shape.iter()) {
        if *d > 0 && *s > 0 {
            max_offset += (d - 1) * s;
        }
    }
    max_offset + 1
}

impl StridedLayout {
    pub fn compute_strides(&self, shape: &Shape) -> Strides {
        let n = shape.len();
        match self {
            StridedLayout::RowMajor => {
                assert!(n >= 1, "RowMajor requires at least 1 dimension");
                let mut strides = vec![0; n];
                strides[n - 1] = 1;
                for i in (0..n - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                Strides::new(&strides)
            }
            StridedLayout::ColMajor => {
                assert!(n >= 2, "ColMajor requires at least 2 dimensions");
                let mut strides = vec![0; n];
                strides[n - 2] = 1;
                strides[n - 1] = shape[n - 2];
                let mut inner = shape[n - 2] * shape[n - 1];
                for i in (0..n - 2).rev() {
                    strides[i] = inner;
                    inner *= shape[i];
                }
                Strides::new(&strides)
            }
            StridedLayout::Explicit(strides) => {
                assert!(
                    strides.len() == n,
                    "Explicit strides must have the same rank as the shape"
                );
                strides.clone().into()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_extent_contiguous_row_major() {
        // Row-major 2x3 → strides (3, 1) → 6 elements covered.
        let shape = Shape::from(vec![2, 3]);
        let strides = Strides::new(&[3, 1]);
        assert_eq!(physical_extent(&shape, &strides), 6);
    }

    #[test]
    fn physical_extent_jumpy_strides_exceed_logical() {
        // 256x256 logical view of a wider 256x512 buffer (stride 512 on dim 0).
        // Last reachable offset is 255*512 + 255*1 = 130815 → +1 = 130816.
        let shape = Shape::from(vec![256, 256]);
        let strides = Strides::new(&[512, 1]);
        assert_eq!(physical_extent(&shape, &strides), 130816);
        // And it strictly exceeds the logical element count.
        assert!(physical_extent(&shape, &strides) > 256 * 256);
    }

    #[test]
    fn physical_extent_broadcast_strides_undercount_logical() {
        // Broadcast dim: stride 0 means every index along that dim shares the
        // same physical offset. A 4x3 tensor broadcasting dim 0 only needs 3
        // elements of physical storage, not 12.
        let shape = Shape::from(vec![4, 3]);
        let strides = Strides::new(&[0, 1]);
        assert_eq!(physical_extent(&shape, &strides), 3);
        // ...and it's less than the logical element count.
        assert!(physical_extent(&shape, &strides) < 4 * 3);
    }

    #[test]
    fn col_major_matrix_steps_a_whole_column_per_column() {
        // 2x3 col-major: rows contiguous, each column 2 elements on → strides (1, 2).
        let strides = StridedLayout::ColMajor.compute_strides(&Shape::from(vec![2, 3]));
        assert_eq!(strides, Strides::new(&[1, 2]));
    }

    #[test]
    fn col_major_batches_step_over_whole_matrices() {
        // Two 2x3 col-major matrices: the batch axis steps 2*3 = 6.
        let shape = Shape::from(vec![2, 2, 3]);
        let strides = StridedLayout::ColMajor.compute_strides(&shape);
        assert_eq!(strides, Strides::new(&[6, 1, 2]));
        assert_eq!(physical_extent(&shape, &strides), 2 * 2 * 3);
    }

    #[test]
    fn col_major_batch_axes_chain_like_row_major() {
        // [2, 3, 4, 5]: a matrix is 4*5 = 20, the inner batch steps 20, the outer 3*20 = 60.
        let strides = StridedLayout::ColMajor.compute_strides(&Shape::from(vec![2, 3, 4, 5]));
        assert_eq!(strides, Strides::new(&[60, 20, 1, 4]));
    }
}
