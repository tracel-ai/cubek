use cubecl::prelude::*;
use cubecl::std::tensor::{
    layout::{Coords1d, Coords2d, Layout, LayoutExpand},
    r#virtual::VirtualTensor,
};
use cubecl::{self as cubecl};
use cubek_matmul::components::global::memory::GlobalMemoryConfig;

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, Clone, Copy)]
pub struct AttentionGlobalLayout {
    rows: u32,
    stride_row: usize,
    columns: u32,
    stride_col: usize,
    batch_offset: usize,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl AttentionGlobalLayout {
    /// Creates a new 2D layout for the `(batch, head)` slice selected by
    /// `batch_index`, the flattened `batch * num_heads + head` index over the
    /// query's extents. `num_heads` is always the query's head count.
    pub fn new<T: Numeric, N: Size, IO: Clone>(
        tensor: &VirtualTensor<T, N, IO>,
        batch_index: u32,
        num_heads: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        let batch = batch_index / num_heads;
        let head = batch_index % num_heads;
        // A broadcast (size-1) batch/head dimension contributes no offset,
        // so a [1, 1, S, S] mask reuses its slice instead of striding past it.
        let batch_part = if tensor.shape(0) > 1 {
            batch as usize * tensor.stride(0)
        } else {
            0usize
        };
        let head_part = if tensor.shape(1) > 1 {
            head as usize * tensor.stride(1)
        } else {
            0usize
        };
        let batch_offset = batch_part + head_part;
        AttentionGlobalLayout {
            rows: tensor.shape(2) as u32,
            stride_row: tensor.stride(2),
            columns: tensor.shape(3) as u32,
            stride_col: tensor.stride(3),
            batch_offset,
            config,
        }
    }
}

#[cube]
impl Layout for AttentionGlobalLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> usize {
        let vector_size = self.config.vector_size.comptime();
        let (row, col) = coords;
        let idx =
            self.batch_offset + row as usize * self.stride_row + col as usize * self.stride_col;

        idx / vector_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.rows, self.columns)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let config = self.config.comptime();
        let (row, col) = pos;

        match (config.check_row_bounds, config.check_col_bounds) {
            (true, true) => row < self.rows && col < self.columns,
            (true, false) => row < self.rows,
            (false, true) => col < self.columns,
            (false, false) => true,
        }
    }
}
