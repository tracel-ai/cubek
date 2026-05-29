//! Inner (physical) layout of a matmul operand — a generalization of
//! [`MatrixLayout`](cubek_std::MatrixLayout).
//!
//! Where `MatrixLayout` says only row- vs col-major of the trailing 2-D matrix,
//! an [`InnerLayout`] also covers batch-major and (recursively) tiled storage.
//! It answers two questions for a logical `(batch, rows, cols)` operand: the
//! physical buffer to allocate, and the launch view that presents the logical
//! shape. Every variant reduces to one mechanism — a [`TiledViewLayout`] over a
//! strided buffer with a [`TileSpec`]: strided variants permute which axis is
//! contiguous (`levels: 0`); tiled variants block the matrix axes (`levels:
//! 1`/`2`). A kernel reads the operand logically through the view, so the layout
//! changes only the memory access pattern, never the result.

use cubecl::{
    Runtime,
    prelude::{TensorArg, TensorBinding},
    std::tensor::layout::tiled_view::{TileSpec, TiledViewLaunch, TiledViewLayout},
};

/// How a logical `(batch, rows, cols)` operand is physically stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerLayout {
    /// `cols` contiguous (standard C order) — `MatrixLayout::RowMajor`.
    RowMajor,
    /// `rows` contiguous within a batch (matrix transposed) — `MatrixLayout::ColMajor`.
    ColMajor,
    // A batch-major / batch-vectorized variant belongs here too, but it needs
    // more than a storage label: the vector width and that *all* operands agree
    // on it (otherwise lanes mix batches). Deferred until there's a kernel that
    // vectorizes along batch — the variants below are deliberately open to it.
    /// Matrix axes blocked into contiguous `tile × tile` sub-tiles (tile = edge,
    /// like `TileInput::tile`), so a leaf is one contiguous block.
    Tiled { tile: usize },
    /// Two nested levels of blocking, with the same convention as
    /// `cubek-tile/tests/tile/recursive.rs`: identical to
    /// `TileInput::split(&[level1; 2]).split(&[level2; 2])`. `level1`/`level2`
    /// are the per-level *split counts* (coarse→fine); the innermost tile is the
    /// leftover `extent / (level1 · level2)`.
    RecursivelyTiled { level1: usize, level2: usize },
}

impl InnerLayout {
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
        match *self {
            InnerLayout::RowMajor | InnerLayout::ColMajor => {
                vec![batch, rows, cols]
            }
            InnerLayout::Tiled { tile } => {
                vec![batch, rows / tile, cols / tile, tile, tile]
            }
            // Block-major, coarse→fine: [batch, grid, grid, level1, level1,
            // leftover, leftover] — the two split counts then the leftover tile,
            // matching `recursive.rs`'s `.split(&[level1;2]).split(&[level2;2])`.
            InnerLayout::RecursivelyTiled { level1, level2 } => {
                let block = level1 * level2;
                vec![
                    batch,
                    level1,
                    level1,
                    level2,
                    level2,
                    rows / block,
                    cols / block,
                ]
            }
        }
    }

    /// Canonical strides that *realize* this layout on a freshly allocated
    /// (contiguous) buffer of [`physical_dims`](Self::physical_dims). Used when
    /// building an operand in a chosen layout (e.g. the layout laboratory);
    /// [`view`](Self::view) itself only preserves whatever strides a binding
    /// already carries.
    pub fn physical_strides(&self, batch: usize, rows: usize, cols: usize) -> Vec<usize> {
        match *self {
            InnerLayout::RowMajor => vec![rows * cols, cols, 1],
            InnerLayout::ColMajor => vec![rows * cols, 1, rows],
            // Tiled buffers carry the layout in their *shape*; strides are plain
            // row-major over those physical dims.
            InnerLayout::Tiled { .. } | InnerLayout::RecursivelyTiled { .. } => {
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
        let spec = match *self {
            // Batch is a passthrough "pre" axis; only the two matrix axes tile.
            InnerLayout::Tiled { .. } => TileSpec {
                start_axis: 1,
                num_tiled: 2,
                levels: 1,
            },
            InnerLayout::RecursivelyTiled { .. } => TileSpec {
                start_axis: 1,
                num_tiled: 2,
                levels: 2,
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
}

impl From<cubek_std::MatrixLayout> for InnerLayout {
    fn from(layout: cubek_std::MatrixLayout) -> Self {
        match layout {
            cubek_std::MatrixLayout::RowMajor => InnerLayout::RowMajor,
            cubek_std::MatrixLayout::ColMajor => InnerLayout::ColMajor,
        }
    }
}
