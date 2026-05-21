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

/// The full layout of a test tensor. Mirrors cubecl's `Metadata` model
/// (strides + `Option<Tiler>`): the base [`StridedLayout`] drives the physical
/// strides; for `Tiled`, the rank-expanded metadata is applied after the
/// buffer is written (see [`cubecl::zspace::metadata::Metadata::to_tiled`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutSpec {
    Strided(StridedLayout),
    /// Splits each axis in `[start_axis, start_axis + tile_shape.len())` into a
    /// grid component and a tile component (`D -> [D/T, T]`), producing
    /// `[Pre-axes, Grid-axes, Tile-axes, Post-axes]` in row-major order.
    Tiled {
        base: StridedLayout,
        start_axis: u8,
        tile_shape: Vec<u16>,
    },
}

impl Default for LayoutSpec {
    fn default() -> Self {
        Self::Strided(StridedLayout::default())
    }
}

impl LayoutSpec {
    /// Compose a base stride layout with a tile.
    pub fn tiled(base: StridedLayout, start_axis: u8, tile_shape: Vec<u16>) -> Self {
        Self::Tiled {
            base,
            start_axis,
            tile_shape,
        }
    }

    /// The base stride layout, with or without tiling.
    pub fn base(&self) -> &StridedLayout {
        match self {
            LayoutSpec::Strided(base) | LayoutSpec::Tiled { base, .. } => base,
        }
    }

    /// `(start_axis, tile_shape)` if this layout is tiled.
    pub fn tile(&self) -> Option<(u8, &[u16])> {
        match self {
            LayoutSpec::Tiled {
                start_axis,
                tile_shape,
                ..
            } => Some((*start_axis, tile_shape)),
            LayoutSpec::Strided(_) => None,
        }
    }

    /// Compute the strides of the base (un-tiled) layout for `shape`.
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
                for i in (0..n - 2).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
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
}
