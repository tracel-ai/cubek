//! Inner (physical) layout of a matmul operand — a generalization of
//! [`MatrixLayout`](cubek_std::MatrixLayout).

use cubecl::{
    Runtime,
    prelude::{TensorArg, TensorBinding},
    std::tensor::layout::tiled_view::{TileSpec, TiledViewLaunch, TiledViewLayout},
};
use cubek_tile::Storage;

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

    /// Physical buffer dims to allocate for a logical `(batch, rows, cols)`
    /// operand. Strided variants store the logical shape (the *strides* carry the
    /// layout); tiled variants expand the matrix axes into `[grid…, tile…]`.
    pub fn physical_dims(&self, batch: usize, rows: usize, cols: usize) -> Vec<usize> {
        match self {
            InnerLayout::RowMajor | InnerLayout::ColMajor => {
                vec![batch, rows, cols]
            }
            // Level-major, coarse→fine: [batch, grid_r, grid_c, …, finest_r,
            // finest_c] — each level contributes both axes' factors, as
            // `TiledLayout` expects (`[pre, grid…, level1…, …]`).
            InnerLayout::Tiled { tiles } => {
                let row_factors = axis_factors(tiles.iter().map(|t| t.0), rows);
                let col_factors = axis_factors(tiles.iter().map(|t| t.1), cols);
                let mut dims = Vec::with_capacity(1 + row_factors.len() * 2);
                dims.push(batch);
                for (r, c) in row_factors.into_iter().zip(col_factors) {
                    dims.push(r);
                    dims.push(c);
                }
                dims
            }
        }
    }

    /// Canonical strides that *realize* this layout on a freshly allocated
    /// (contiguous) buffer of [`physical_dims`](Self::physical_dims). Used when
    /// building an operand in a chosen layout (e.g. the layout laboratory);
    /// [`view`](Self::view) itself only preserves whatever strides a binding
    /// already carries.
    pub fn physical_strides(&self, batch: usize, rows: usize, cols: usize) -> Vec<usize> {
        match self {
            InnerLayout::RowMajor => vec![rows * cols, cols, 1],
            InnerLayout::ColMajor => vec![rows * cols, 1, rows],
            // Tiled buffers carry the layout in their *shape*; strides are plain
            // row-major over those physical dims.
            InnerLayout::Tiled { .. } => {
                let dims = self.physical_dims(batch, rows, cols);
                let mut strides = vec![1usize; dims.len()];
                for i in (0..dims.len() - 1).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1];
                }
                strides
            }
        }
    }

    /// A launch view over `binding` presenting the logical `(batch, rows, cols)`.
    /// The binding's strides are *preserved* — they (or its physical shape, for
    /// tiled) already encode the layout — so this is correct for any incoming
    /// layout. Strided variants are reshaped to a `(batch, rows, cols)` view;
    /// tiled variants keep their physical `[grid…, tile…]` shape and a
    /// [`TileSpec`] reads the tile sizes from it.
    pub fn view<R: Runtime>(
        &self,
        mut binding: TensorBinding<R>,
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> TiledViewLaunch<R> {
        let spec = match self {
            // Batch is a passthrough "pre" axis; only the two matrix axes tile,
            // with one nesting level per `tiles` entry.
            InnerLayout::Tiled { tiles } => TileSpec {
                start_axis: 1,
                num_tiled: 2,
                levels: tiles.len(),
            },
            // Strided: reshape to (batch, rows, cols), preserving the operand's
            // own strides (batch stride taken from the binding, not assumed).
            _ => {
                let strides = binding.strides.to_vec();
                let n = strides.len();
                let batch_stride = if n >= 3 { strides[n - 3] } else { rows * cols };
                binding.shape = [batch, rows, cols][..].into();
                binding.strides = [batch_stride, strides[n - 2], strides[n - 1]][..].into();
                TileSpec {
                    start_axis: 0,
                    num_tiled: 3,
                    levels: 0,
                }
            }
        };

        let arg: TensorArg<R> = binding.into_tensor_arg();
        TiledViewLaunch::new_tensor::<TiledViewLayout>(arg, spec)
    }

    /// The raw [`TensorArg`] (operand strides preserved) plus the tensor's physical
    /// [`Storage`] that `Tile::from_tensor` needs in-kernel. Tiled keeps its physical
    /// `[batch, grid…, tile…]` buffer (batch passthrough, `start_axis = 1`); strided
    /// reshapes to `(batch, rows, cols)` (`start_axis = 0, levels = 0`).
    pub fn tensor_arg<R: Runtime>(
        &self,
        mut binding: TensorBinding<R>,
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> (TensorArg<R>, Storage) {
        match self {
            InnerLayout::Tiled { tiles } => (
                binding.into_tensor_arg(),
                Storage::passthrough(1, tiles.len()),
            ),
            _ => {
                let strides = binding.strides.to_vec();
                let n = strides.len();
                let batch_stride = if n >= 3 { strides[n - 3] } else { rows * cols };
                binding.shape = [batch, rows, cols][..].into();
                binding.strides = [batch_stride, strides[n - 2], strides[n - 1]][..].into();
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
