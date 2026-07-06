//! Test-only physical-layout enum for the cpu_gemm layout laboratory: a compact
//! `RowMajor`/`ColMajor`/`Tiled` description that builds synthetic operands (their physical dims +
//! strides) and converts to the real [`ConcreteLayout`]. Production carries no such enum — it reads
//! the storage-tiling depth + strides straight off each binding — so this lives with the tests.

use cubecl::{
    Runtime,
    prelude::{TensorArg, TensorBinding},
};
use cubek_matmul::definition::MatmulSetupError;
use cubek_std::MatrixLayout;
use cubek_tile::{Axis, ConcreteLayout, PhysicalAxis, Storage};

/// How a logical `(batch, rows, cols)` operand is physically stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InnerLayout {
    /// `cols` contiguous (standard C order) — `MatrixLayout::RowMajor`.
    RowMajor,
    /// `rows` contiguous within a batch (matrix transposed) — `MatrixLayout::ColMajor`.
    ColMajor,
    /// Matrix axes blocked into nested, contiguous sub-tiles. Each entry is one
    /// nesting level's `(row_edge, col_edge)`, outer→inner — so tiles may be
    /// rectangular and arbitrarily deep:
    /// - `[(4, 4)]` — plain `4 × 4` blocks (a leaf is one contiguous block).
    /// - `[(8, 4)]` — rectangular `8 × 4` blocks.
    /// - `[(4, 4), (2, 2)]` — `4 × 4` blocks each split into `2 × 2`.
    Tiled { tiles: Vec<(usize, usize)> },
}

/// Per-axis mixed-radix factors `[grid, between-levels…, finest tile]` for an
/// axis of length `extent` whose nesting edges (outer→inner) are `edges`.
fn axis_factors(edges: impl IntoIterator<Item = usize>, extent: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut prev = extent;
    for edge in edges {
        factors.push(prev / edge);
        prev = edge;
    }
    factors.push(prev);
    factors
}

impl InnerLayout {
    /// Storage-tiling depth: `levels` nested `[grid…, leaf]` splits per matrix axis (`0` strided).
    pub fn levels(&self) -> usize {
        match self {
            InnerLayout::RowMajor | InnerLayout::ColMajor => 0,
            InnerLayout::Tiled { tiles } => tiles.len(),
        }
    }

    /// Convenience: a single level of square `edge × edge` blocks.
    pub fn square_tiled(edge: usize) -> Self {
        InnerLayout::Tiled {
            tiles: vec![(edge, edge)],
        }
    }

    /// Detect the (strided) inner layout of a plain tensor from its strides: whichever of the
    /// trailing two matrix axes is contiguous. Tiled layouts aren't expressible as plain strides.
    pub fn from_shape_and_strides(
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, MatmulSetupError> {
        Ok(MatrixLayout::from_shape_and_strides(shape, strides, None)?.into())
    }

    /// The per-operand [`ConcreteLayout`] this imposes on the matrix axes `[row, col]`, in physical
    /// (major-to-minor) order — for layout-request matching: row-major makes `col` innermost,
    /// col-major `row`, a tiled layout expands each matrix axis into its `[grid…, leaf]` fragments.
    pub fn to_concrete(
        &self,
        matrix: [Axis; 2],
        num_rows: usize,
        num_cols: usize,
    ) -> ConcreteLayout {
        let [row, col] = matrix;
        match self {
            InnerLayout::RowMajor => ConcreteLayout::new(&[
                PhysicalAxis::new(row, num_rows),
                PhysicalAxis::new(col, num_cols),
            ]),
            InnerLayout::ColMajor => ConcreteLayout::new(&[
                PhysicalAxis::new(col, num_cols),
                PhysicalAxis::new(row, num_rows),
            ]),
            InnerLayout::Tiled { tiles } => {
                let row_factors = axis_factors(tiles.iter().map(|t| t.0), num_rows);
                let col_factors = axis_factors(tiles.iter().map(|t| t.1), num_cols);
                let mut axes = Vec::with_capacity(row_factors.len() * 2);
                for (r, c) in row_factors.into_iter().zip(col_factors) {
                    axes.push(PhysicalAxis::new(row, r));
                    axes.push(PhysicalAxis::new(col, c));
                }
                ConcreteLayout::new(&axes)
            }
        }
    }

    /// Physical buffer dims to allocate for a logical `(batches, rows, cols)` operand. Strided
    /// variants store the logical shape (the *strides* carry the layout); tiled variants expand the
    /// matrix axes into `[grid…, tile…]`.
    pub fn physical_dims(&self, batches: &[usize], rows: usize, cols: usize) -> Vec<usize> {
        match self {
            InnerLayout::RowMajor | InnerLayout::ColMajor => {
                let mut dims = batches.to_vec();
                dims.extend([rows, cols]);
                dims
            }
            InnerLayout::Tiled { tiles } => {
                let row_factors = axis_factors(tiles.iter().map(|t| t.0), rows);
                let col_factors = axis_factors(tiles.iter().map(|t| t.1), cols);
                let mut dims = batches.to_vec();
                dims.reserve(row_factors.len() * 2);
                for (r, c) in row_factors.into_iter().zip(col_factors) {
                    dims.push(r);
                    dims.push(c);
                }
                dims
            }
        }
    }

    /// Canonical strides that *realize* this layout on a freshly allocated (contiguous) buffer of
    /// [`physical_dims`](Self::physical_dims).
    pub fn physical_strides(&self, batches: &[usize], rows: usize, cols: usize) -> Vec<usize> {
        fn row_major_strides(dims: &[usize]) -> Vec<usize> {
            let mut strides = vec![1usize; dims.len()];
            for i in (0..dims.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            strides
        }
        match self {
            InnerLayout::RowMajor => {
                let mut strides = row_major_strides(batches)
                    .iter()
                    .map(|s| s * rows * cols)
                    .collect::<Vec<_>>();
                strides.extend([cols, 1]);
                strides
            }
            InnerLayout::ColMajor => {
                let mut strides = row_major_strides(batches)
                    .iter()
                    .map(|s| s * rows * cols)
                    .collect::<Vec<_>>();
                strides.extend([1, rows]);
                strides
            }
            InnerLayout::Tiled { .. } => {
                row_major_strides(&self.physical_dims(batches, rows, cols))
            }
        }
    }

    /// The raw [`TensorArg`] (strides preserved) plus the physical [`Storage`] that
    /// `TileArgLaunch::strided` turns into a `TiledViewLayout` view. `vector_size > 1` lines the
    /// innermost (`cols`) axis (only valid for a row-major operand; tiled passes `1`).
    pub fn tensor_arg<R: Runtime>(
        &self,
        mut binding: TensorBinding<R>,
        vector_size: usize,
    ) -> (TensorArg<R>, Storage) {
        match self {
            InnerLayout::Tiled { tiles } => {
                let levels = tiles.len();
                let num_batch = binding.shape.len() - 2 * (levels + 1);
                (
                    binding.into_tensor_arg(),
                    Storage::passthrough(num_batch, levels),
                )
            }
            _ => {
                let n = binding.strides.len();
                let mut shape = binding.shape.to_vec();
                let mut strides = binding.strides.to_vec();
                shape[n - 1] /= vector_size;
                for s in &mut strides[..n - 1] {
                    *s /= vector_size;
                }
                binding.shape = shape[..].into();
                binding.strides = strides[..].into();
                (binding.into_tensor_arg(), Storage::passthrough(0, 0))
            }
        }
    }
}

impl From<MatrixLayout> for InnerLayout {
    fn from(layout: MatrixLayout) -> Self {
        match layout {
            MatrixLayout::RowMajor => InnerLayout::RowMajor,
            MatrixLayout::ColMajor => InnerLayout::ColMajor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduces_row_major_from_contiguous_cols() {
        let l = InnerLayout::from_shape_and_strides(&[2, 4, 8], &[32, 8, 1]).unwrap();
        assert_eq!(l, InnerLayout::RowMajor);
    }

    #[test]
    fn deduces_col_major_from_contiguous_rows() {
        let l = InnerLayout::from_shape_and_strides(&[2, 4, 8], &[32, 1, 4]).unwrap();
        assert_eq!(l, InnerLayout::ColMajor);
    }

    #[test]
    fn rejects_strided_contiguous_in_neither_axis() {
        assert!(InnerLayout::from_shape_and_strides(&[2, 4, 8], &[64, 16, 2]).is_err());
    }
}
