//! Inner (physical) layout of a matmul operand — a generalization of
//! [`MatrixLayout`](cubek_std::MatrixLayout).

use cubecl::{
    Runtime,
    prelude::{TensorArg, TensorBinding},
};
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
    ///
    /// Each level's edge must divide the one enclosing it, and the outermost must
    /// divide the axis extent. The grid (count of outermost blocks) is the
    /// leftover quotient, so the same value applies to any matrix size.
    Tiled { tiles: Vec<(usize, usize)> },
}

/// Per-axis mixed-radix factors `[grid, between-levels…, finest tile]` for an
/// axis of length `extent` whose nesting edges (outer→inner) are `edges`. The
/// product telescopes back to `extent`; the finest factor is the innermost edge.
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
    /// Convenience: a single level of square `edge × edge` blocks.
    pub fn square_tiled(edge: usize) -> Self {
        InnerLayout::Tiled {
            tiles: vec![(edge, edge)],
        }
    }

    /// Detect the (strided) inner layout of a plain tensor from its strides:
    /// whichever of the trailing two matrix axes is contiguous. Tiled layouts
    /// aren't expressible as plain strides, so a standard binding only ever
    /// resolves to a strided variant.
    pub fn from_shape_and_strides(shape: &[usize], strides: &[usize]) -> Self {
        let n = shape.len();
        if strides[n - 2] == 1 && strides[n - 1] >= shape[n - 2] {
            InnerLayout::ColMajor
        } else {
            InnerLayout::RowMajor
        }
    }

    /// The per-operand [`ConcreteLayout`] this imposes on the matrix axes `[row, col]`:
    /// row-major makes `col` innermost, col-major `row`, a tiled layout expands each matrix
    /// axis into its `[grid…, leaf]` fragments (level-major, leaf innermost) just like the
    /// physical buffer. Batch axes are layout-irrelevant, so only the matrix is described.
    /// Temporary bridge while `InnerLayout` converges onto `ConcreteLayout`.
    #[allow(dead_code)]
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
            // Level-major `[grid_r, grid_c, …, leaf_r, leaf_c]`, mirroring `physical_dims`: a
            // tiled axis is repeated, one fragment per level, so the leaf lands innermost.
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

    /// Physical buffer dims to allocate for a logical `(batches, rows, cols)`
    /// operand. `batches` is the per-dimension batch shape (one entry per batch axis;
    /// empty for an unbatched operand). Strided variants store the logical shape (the
    /// *strides* carry the layout); tiled variants expand the matrix axes into
    /// `[grid…, tile…]`.
    pub fn physical_dims(&self, batches: &[usize], rows: usize, cols: usize) -> Vec<usize> {
        match self {
            InnerLayout::RowMajor | InnerLayout::ColMajor => {
                let mut dims = batches.to_vec();
                dims.extend([rows, cols]);
                dims
            }
            // Level-major, coarse→fine: [batches…, grid_r, grid_c, …, finest_r,
            // finest_c] — each level contributes both axes' factors, as
            // `TiledLayout` expects (`[pre, grid…, level1…, …]`).
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

    /// Recover the logical `(batches, rows, cols)` from a physical shape in this layout
    /// — the inverse of [`physical_dims`](Self::physical_dims). `batches` is the
    /// per-dimension batch shape (the leading dims). Strided variants store the logical
    /// shape directly; tiled variants fold the per-level `(row, col)` factors back into
    /// `rows`/`cols`, the leading `physical.len() - 2·(levels)` dims being the batch.
    pub fn logical_dims(&self, physical: &[usize]) -> (Vec<usize>, usize, usize) {
        match self {
            InnerLayout::RowMajor | InnerLayout::ColMajor => {
                let n = physical.len();
                (physical[..n - 2].to_vec(), physical[n - 2], physical[n - 1])
            }
            // [batches…, r0, c0, r1, c1, …]: the trailing `2·(tiles+1)` dims are the
            // matrix's per-level row/col factors; everything before them is the batch.
            InnerLayout::Tiled { tiles } => {
                let matrix = 2 * (tiles.len() + 1);
                let split = physical.len() - matrix;
                let (mut rows, mut cols) = (1, 1);
                for (i, &d) in physical[split..].iter().enumerate() {
                    if i % 2 == 0 {
                        rows *= d;
                    } else {
                        cols *= d;
                    }
                }
                (physical[..split].to_vec(), rows, cols)
            }
        }
    }

    /// Canonical strides that *realize* this layout on a freshly allocated
    /// (contiguous) buffer of [`physical_dims`](Self::physical_dims). Used when
    /// building an operand in a chosen layout (e.g. the layout laboratory);
    /// [`view`](Self::view) itself only preserves whatever strides a binding
    /// already carries.
    pub fn physical_strides(&self, batches: &[usize], rows: usize, cols: usize) -> Vec<usize> {
        // Row-major (contiguous) strides over an arbitrary shape.
        fn row_major_strides(dims: &[usize]) -> Vec<usize> {
            let mut strides = vec![1usize; dims.len()];
            for i in (0..dims.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            strides
        }
        match self {
            // Batch is row-major over `rows·cols`-sized matrices; the matrix itself is
            // contiguous (RowMajor) or transposed (ColMajor).
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
            // Tiled buffers carry the layout in their *shape*; strides are plain
            // row-major over those physical dims.
            InnerLayout::Tiled { .. } => {
                row_major_strides(&self.physical_dims(batches, rows, cols))
            }
        }
    }

    /// The raw [`TensorArg`] (operand strides preserved) plus the tensor's physical
    /// [`Storage`] that `Tile::from_tensor` needs in-kernel. The batch count is read off
    /// the binding's rank: every leading dim before the matrix is a batch axis, so a
    /// broadcast operand simply arrives with the size-1 batch dims already squeezed out
    /// (its omitted axes). Tiled keeps its physical `[batches…, grid…, tile…]` buffer
    /// (batch passthrough, `start_axis = num_batch`); strided is a plain
    /// `[batches…, rows, cols]` dot (`levels = 0`).
    ///
    /// `vector_size > 1` lines the innermost (`cols`) axis: its shape and every
    /// non-contiguous stride are divided by the line size, so a kernel reading
    /// `Vector<E, vector_size>` lands on contiguous lines. Only valid when `cols` is
    /// contiguous (a row-major operand); tiled operands must pass `vector_size = 1`.
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
                // Re-line the buffer as `Vector<E, v>`: the contiguous innermost stride
                // stays 1, every coarser stride and the `cols` extent shrink by `v`.
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

impl From<cubek_std::MatrixLayout> for InnerLayout {
    fn from(layout: cubek_std::MatrixLayout) -> Self {
        match layout {
            cubek_std::MatrixLayout::RowMajor => InnerLayout::RowMajor,
            cubek_std::MatrixLayout::ColMajor => InnerLayout::ColMajor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_tile::{AxisSet, Constraint, Facet, LayoutRequest};

    const A: Axis = Axis(0); // matrix row axis
    const B: Axis = Axis(1); // matrix col axis

    fn wants_innermost(axis: Axis) -> LayoutRequest {
        LayoutRequest::new().with(Constraint::required(Facet::Innermost(AxisSet::one(axis))))
    }

    #[test]
    fn row_major_puts_col_innermost() {
        let layout = InnerLayout::RowMajor.to_concrete([A, B], 8, 4);
        assert!(wants_innermost(B).feasible(&layout));
        assert!(!wants_innermost(A).feasible(&layout));
    }

    #[test]
    fn col_major_puts_row_innermost() {
        let layout = InnerLayout::ColMajor.to_concrete([A, B], 8, 4);
        assert!(wants_innermost(A).feasible(&layout));
        assert!(!wants_innermost(B).feasible(&layout));
    }

    #[test]
    fn tiled_keeps_col_innermost_and_records_tiling() {
        let layout = InnerLayout::square_tiled(4).to_concrete([A, B], 16, 16);
        assert!(wants_innermost(B).feasible(&layout));
        let wants_tiled =
            LayoutRequest::new().with(Constraint::required(Facet::Tiled { axis: B, edge: 4 }));
        assert!(wants_tiled.feasible(&layout));
    }
}
