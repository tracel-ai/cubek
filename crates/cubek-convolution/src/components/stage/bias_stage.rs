use cubecl::prelude::*;
use cubek_matmul::components::{
    stage::{
        LoadStageFamily, Stage, StageFamily, StageMemoryConfig, TilingLayout, as_swizzle_object,
    },
    tile::{StridedTile, io::Strided},
};

use cubecl::std::{Swizzle, tensor::layout::Coords2d, type_size};

use crate::components::stage::reader::BiasTilingLayout;

pub struct BiasStageFamily;

impl StageFamily for BiasStageFamily {
    type TileKind = Strided;

    type Stage<ES: Numeric, T: TilingLayout> = BiasStageMemory<ES>;
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct BiasStageMemory<ES: Numeric> {
    /// Underlying shared memory
    pub smem: SharedMemory<Line<ES>>,
    /// Swizzling of the shared memory, if any
    pub swizzle: Swizzle,
    buffer_index: u32,

    #[cube(comptime)]
    stage_size: u32,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl<ES: Numeric> BiasStageMemory<ES> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(#[comptime] config: StageMemoryConfig) -> BiasStageMemory<ES> {
        Self::new_aligned(type_size::<ES>(config.line_size as usize), config)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> BiasStageMemory<ES> {
        let line_size = config.line_size as usize;
        let swizzle = as_swizzle_object(config.swizzle);
        let swizzle_align = swizzle.repeats_after();
        let align = comptime![Ord::max(alignment, swizzle_align as usize)];
        let type_size = type_size::<ES>(line_size).comptime();

        let stage_size_bytes =
            config.elements_per_stage_along_contiguous_dim() as usize * type_size;
        // Ensure all stages are aligned properly
        let stage_size = stage_size_bytes.next_multiple_of(align) / type_size / line_size;

        let smem =
            SharedMemory::new_aligned(config.num_stages as usize * stage_size, line_size, align);

        BiasStageMemory::<ES> {
            smem,
            swizzle,
            stage_size: stage_size as u32,
            config,
            buffer_index: 0u32,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        BiasStageMemory::<ES> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: buffer_idx,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile(&self, tile: Coords2d) -> StridedTile<ES> {
        BiasTilingLayout::get_tile::<ES>(self, tile, self.config)
    }

    /// Get the tile at position (row, col)
    pub fn get_tile_mut(&self, tile: Coords2d) -> StridedTile<ES, ReadWrite> {
        let tile = self.get_tile(tile);
        StridedTile::<ES, ReadWrite> {
            stage: tile.stage.as_mut_unchecked(),
            start: tile.start,
            end: tile.end,
            stride: tile.stride,
            swizzle: tile.swizzle,
            layout: tile.layout,
            line_size: tile.line_size,
        }
    }

    /// Return the whole stage as a slice, for reading
    pub fn as_slice(&self, #[comptime] line_size: LineSize) -> Slice<Line<ES>> {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem
            .slice(stage_offset, stage_offset + self.stage_size as usize)
            .with_line_size(line_size)
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self, #[comptime] line_size: LineSize) -> SliceMut<Line<ES>> {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem
            .slice_mut(stage_offset, stage_offset + self.stage_size as usize)
            .with_line_size(line_size)
    }

    /// Frees the shared memory for reuse, if possible on the target runtime.
    ///
    /// # Safety
    /// *Must* be used in uniform control flow
    /// *Must not* have any dangling references to this shared memory
    pub unsafe fn free(self) {
        unsafe { self.smem.free() };
    }
}

#[cube]
impl<ES: Numeric> Stage<ES, ReadOnly> for BiasStageMemory<ES> {
    type TileKind = Strided;

    fn tile(this: &Self, tile: Coords2d) -> StridedTile<ES> {
        this.get_tile(tile)
    }
}

#[cube]
impl LoadStageFamily<ReadOnly> for BiasStageFamily {
    fn create<ES: Numeric, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, T> {
        BiasStageMemory::new_aligned(alignment, config)
    }

    fn with_buffer_index<ES: Numeric, T: TilingLayout>(
        stage: &Self::Stage<ES, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, T> {
        stage.with_buffer_index(buffer_index)
    }

    fn free<ES: Numeric, T: TilingLayout>(stage: &Self::Stage<ES, T>) {
        unsafe { stage.free() };
    }
}
