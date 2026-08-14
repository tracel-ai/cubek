use cubecl::prelude::*;
use cubecl::std::{
    FastDivmod,
    tensor::layout::{
        Coords1d, Coords2d, Layout, LayoutExpand, VirtualLayout, VirtualLayoutLaunch,
    },
};
use cubecl::zspace::Shape;
use cubecl_common::quant::scheme::{BlockSize, QuantScheme};
use cubek_std::MatrixLayout;

use crate::{
    definition::MatmulProblem,
    {args::BatchedCoords, components::global::memory::GlobalMemoryConfig},
};

/// Global layout that uses the last two dimensions and ignores all others.
/// Same rules as cubek-tile's `TmaDynLayout` (dyn coords); keep the two in step.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct SimpleTmaGlobalLayout {
    #[cube(comptime)]
    transposed: bool,
    shape: BatchedCoords,
}

#[cube]
impl SimpleTmaGlobalLayout {
    /// Creates a new 2D layout with the batch set to `nth_batch`.
    pub fn new(shape: BatchedCoords, #[comptime] layout: MatrixLayout) -> Self {
        let transposed = comptime![matches!(layout, MatrixLayout::ColMajor)];
        SimpleTmaGlobalLayout { shape, transposed }
    }
}

#[cube]
impl Layout for SimpleTmaGlobalLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = BatchedCoords;

    fn to_source_pos(&self, coords: Self::Coordinates) -> BatchedCoords {
        let (batch, row, col) = coords;
        // Don't care if it's actually broadcast, setting batch to 0 is fine either way
        let batch = select(self.shape.0 == 1, 0, batch);
        // Tensor maps are required to have a stride of 1 on the last dim, so their shape is
        // transposed for col-major matrices. Need to compensate by swapping the coordinates.
        if self.transposed.comptime() {
            (batch, col, row)
        } else {
            (batch, row, col)
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (BatchedCoords, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // No need to bounds check TMA loads
        true.runtime()
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct GlobalLayoutConfig {
    pub matrix_layout: MatrixLayout,
    pub check_row_bounds: bool,
    pub check_col_bounds: bool,
}

impl From<GlobalMemoryConfig> for GlobalLayoutConfig {
    fn from(gmem_config: GlobalMemoryConfig) -> Self {
        gmem_config.as_global_layout_config()
    }
}

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
pub struct GlobalLayout {
    batch_layout: VirtualLayout<Coords1d, Coords1d>,
    rows: u32,
    cols: u32,

    stride_row: usize,
    stride_col: usize,

    #[cube(comptime)]
    vector_size: VectorSize,
    #[cube(comptime)]
    packing: u32,
    #[cube(comptime)]
    config: GlobalLayoutConfig,
}

#[cube]
impl GlobalLayout {
    /// Create a new batched global layout. `batch_shape` should be based on the output shape.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_layout: VirtualLayout<Coords1d, Coords1d>,
        shape_row: u32,
        shape_col: u32,
        stride_row: usize,
        stride_col: usize,
        #[comptime] vector_size: VectorSize,
        #[comptime] packing: u32,
        #[comptime] config: GlobalLayoutConfig,
    ) -> Self {
        GlobalLayout {
            batch_layout,
            rows: shape_row,
            cols: shape_col,
            stride_row,
            stride_col,
            vector_size,
            packing,
            config,
        }
    }
}

#[cube]
impl Layout for GlobalLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> usize {
        let (batch, row, col) = coords;
        let batch_offs = self.batch_layout.to_source_pos(batch);

        let (row, col) = match self.config.matrix_layout.comptime() {
            MatrixLayout::RowMajor => (row, col / self.packing),
            MatrixLayout::ColMajor => (row / self.packing, col),
        };

        let idx = batch_offs + row as usize * self.stride_row + col as usize * self.stride_col;

        idx / self.vector_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX.runtime() as usize, self.rows, self.cols)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let config = self.config.comptime();
        let (_, row, col) = pos;

        match (config.check_row_bounds, config.check_col_bounds) {
            (true, true) => row < self.rows && col < self.cols,
            (true, false) => row < self.rows,
            (false, true) => col < self.cols,
            (false, false) => true,
        }
    }
}

impl<R: Runtime> GlobalLayoutLaunch<R> {
    pub fn from_handle(
        handle: &TensorBinding<R>,
        vector_size: VectorSize,
        config: GlobalLayoutConfig,
    ) -> Self {
        let rank = handle.shape.len();
        let rows = handle.shape[rank - 2];
        let cols = handle.shape[rank - 1];
        let stride_row = handle.strides[rank - 2];
        let stride_col = handle.strides[rank - 1];

        GlobalLayoutLaunch::new(
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            rows as u32,
            cols as u32,
            stride_row,
            stride_col,
            vector_size,
            1,
            config,
        )
    }

    pub fn from_handle_batched(
        handle: &TensorBinding<R>,
        problem: &MatmulProblem,
        vector_size: VectorSize,
        config: GlobalLayoutConfig,
    ) -> Self {
        let rank = handle.shape.len();
        let rows = handle.shape[rank - 2];
        let cols = handle.shape[rank - 1];
        let stride_row = handle.strides[rank - 2];
        let stride_col = handle.strides[rank - 1];

        let batch_layout = BatchLayoutLaunch::from_handle(handle, problem);

        GlobalLayoutLaunch::new(
            VirtualLayoutLaunch::new::<BatchLayout>(batch_layout),
            rows as u32,
            cols as u32,
            stride_row,
            stride_col,
            vector_size,
            1,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_quantized_handle(
        values: &TensorBinding<R>,
        scales: &TensorBinding<R>,
        shape: &Shape,
        problem: &MatmulProblem,
        scheme: QuantScheme,
        vector_size: VectorSize,
        config: GlobalLayoutConfig,
    ) -> (GlobalLayoutLaunch<R>, GlobalScaleLayoutLaunch<R>) {
        let rank = values.shape.len();
        let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
        let values_layout = {
            let (stride_row, stride_col) = (values.strides[rank - 2], values.strides[rank - 1]);

            let batch_layout = BatchLayoutLaunch::from_handle(values, problem);

            // The packing axis follows the scheme packed dim, not config.matrix_layout:
            // GEMV reads the RHS ColMajor while the packed buffer stays RowMajor.
            let values_config = GlobalLayoutConfig {
                matrix_layout: match scheme.packing_dim() {
                    Some(1) => MatrixLayout::ColMajor,
                    _ => MatrixLayout::RowMajor,
                },
                ..config
            };

            GlobalLayoutLaunch::new(
                VirtualLayoutLaunch::new::<BatchLayout>(batch_layout),
                rows as u32,
                cols as u32,
                stride_row,
                stride_col,
                vector_size / scheme.num_quants(),
                scheme.num_quants() as u32,
                values_config,
            )
        };

        let scales_layout = {
            let shape = (rows as u32, cols as u32);

            if scheme.num_levels() > 1 {
                unimplemented!(
                    "two-level quantization is not supported by the quantized matmul, got {scheme:?}"
                );
            }

            // Whole-tensor granularity covers both axes, the case `addressing_block` drops.
            let [block_row, block_col] = match scheme.block_size() {
                None => [BlockSize::FULL; 2],
                Some(block) => block.as_dim(),
            };
            let block_size = (
                addressing_block(block_row, rows),
                addressing_block(block_col, cols),
            );
            // Broadcast batch strides are zeroed by `BatchLayoutLaunch::from_handle`, so this asks
            // whether any batch of scales is distinct from the first.
            let batched = scales.shape[..scales.shape.len().saturating_sub(2)]
                .iter()
                .any(|&dim| dim > 1);

            // Scales are never vectorized because we require that `block_size >= vector_size * num_quants`.
            let scales_layout = GlobalLayoutLaunch::from_handle_batched(scales, problem, 1, config);
            GlobalScaleLayoutLaunch::new(shape, scales_layout, block_size, batched)
        };

        (values_layout, scales_layout)
    }
}

/// Keeps a block extent only where it still tells two scales apart. [`BlockSize::FULL`] covers the
/// axis, and so does any extent reaching it, which is also what keeps `FULL`'s zero out of the
/// division below.
fn addressing_block(block: u8, extent: usize) -> Option<u32> {
    (block != BlockSize::FULL && (block as usize) < extent).then_some(block as u32)
}

#[derive(CubeType, CubeLaunch)]
pub struct BatchLayout {
    batch_shape: Sequence<FastDivmod<u32>>,
    batch_strides: Sequence<usize>,
}

#[cube]
impl BatchLayout {
    pub fn new(batch_strides: Sequence<usize>, batch_shape: Sequence<FastDivmod<u32>>) -> Self {
        BatchLayout {
            batch_shape,
            batch_strides,
        }
    }
}

#[cube]
impl Layout for BatchLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut batch = pos as u32;
        let mut batch_offs = 0;
        let batch_shape = self.batch_shape.reversed();
        let batch_strides = self.batch_strides.reversed();

        #[unroll]
        for i in 0..batch_shape.len() {
            let (rem, local_pos) = batch_shape[i].div_mod(batch);
            batch = rem;
            batch_offs += local_pos as usize * batch_strides[i];
        }

        batch_offs
    }

    #[allow(clippy::legacy_numeric_constants)]
    fn shape(&self) -> Self::Coordinates {
        usize::max_value()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

/// Layout that passed through the coordinates with no checks or modification.
#[derive(CubeType, CubeLaunch)]
pub struct NoopLayout {}

#[cube]
impl NoopLayout {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        NoopLayout {}
    }
}

#[cube]
impl Layout for NoopLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        pos
    }

    #[allow(clippy::legacy_numeric_constants)]
    fn shape(&self) -> Self::Coordinates {
        usize::max_value()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

impl<R: Runtime> BatchLayoutLaunch<R> {
    pub fn from_handle(handle: &TensorBinding<R>, problem: &MatmulProblem) -> Self {
        let rank = handle.shape.len();
        let batch_shape = problem
            .out_batches
            .iter()
            .map(|shape| *shape as u32)
            .collect();
        let batch_strides = handle.strides[..rank - 2]
            .iter()
            .zip(&handle.shape[..rank - 2])
            .map(|(stride, shape)| if *shape == 1 { 0 } else { *stride })
            .collect();
        BatchLayoutLaunch::new(batch_shape, batch_strides)
    }
}

/// Maps a value coordinate to the flat index of its block's scale.
///
/// Per-tensor is the degenerate case where the block covers both axes and the scales do not vary
/// across batches: every term folds away at comptime and a read is a constant-index broadcast, so
/// it needs no layout of its own.
#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
pub struct GlobalScaleLayout {
    shape: Coords2d,
    scales_layout: GlobalLayout,
    /// Per-axis block extent, `None` on an axis its block covers, whose quotient is always `0`.
    /// Never the axis extent: holding that would key the kernel on the matrix shape.
    #[cube(comptime)]
    block_size: (Option<u32>, Option<u32>),
    /// Whether the scales vary across batches. They do not when the handle is broadcast over them,
    /// and the batch term is then the one part no block can rule out on its own.
    #[cube(comptime)]
    batched: bool,
}

#[cube]
impl GlobalScaleLayout {
    pub fn new(
        shape: Coords2d,
        scales_layout: GlobalLayout,
        #[comptime] block_size: (Option<u32>, Option<u32>),
        #[comptime] batched: bool,
    ) -> Self {
        GlobalScaleLayout {
            shape,
            scales_layout,
            block_size,
            batched,
        }
    }

    /// Whether any coordinate can move the index off `0`, which decides whether an address is
    /// worth computing at all.
    fn addresses(&self) -> comptime_type!(bool) {
        comptime!(self.batched || self.block_size.0.is_some() || self.block_size.1.is_some())
    }

    /// The axis coordinate's block index, dropped at comptime on an axis holding a single scale
    /// rather than dividing a runtime coordinate that can only answer `0`.
    fn block_index(&self, pos: u32, #[comptime] block: Option<u32>) -> u32 {
        if comptime!(block.is_some()) {
            pos / comptime!(block.unwrap())
        } else {
            0u32.runtime()
        }
    }
}

#[cube]
impl Layout for GlobalScaleLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> usize {
        let addresses = self.addresses();
        if comptime!(!addresses) {
            // One scale for the whole tensor, so no coordinate can reach another.
            0usize.runtime()
        } else {
            let (batch, row, col) = coords;
            let row = self.block_index(row, comptime!(self.block_size.0));
            let col = self.block_index(col, comptime!(self.block_size.1));
            self.scales_layout.to_source_pos((batch, row, col))
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX.runtime() as usize, self.shape.0, self.shape.1)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let addresses = self.addresses();
        if comptime!(!addresses) {
            // The single scale sits at index 0, which no coordinate can leave.
            true.runtime()
        } else {
            let (_, row, col) = pos;
            let config = &self.scales_layout.config.comptime();
            let (rows, cols) = self.shape;

            match (config.check_row_bounds, config.check_col_bounds) {
                (true, true) => row < rows && col < cols,
                (true, false) => row < rows,
                (false, true) => col < cols,
                (false, false) => true,
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct Transpose<Inner: Layout + LaunchArg> {
    inner: Inner,
}

#[cube]
impl<Inner: Layout + LaunchArg> Transpose<Inner> {
    pub fn new(inner: Inner) -> Self {
        Transpose::<Inner> { inner }
    }
}

#[cube]
impl<Inner: Layout<Coordinates = BatchedCoords> + LaunchArg> Layout for Transpose<Inner> {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Inner::SourceCoordinates;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (batch, row, col) = pos;
        self.inner.to_source_pos((batch, col, row))
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (batch, row, col) = pos;
        self.inner.is_in_bounds((batch, col, row))
    }

    fn shape(&self) -> Self::Coordinates {
        let (batches, rows, cols) = self.inner.shape();
        (batches, cols, rows)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
