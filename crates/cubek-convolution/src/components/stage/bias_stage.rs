use cubecl::prelude::*;
use cubek_matmul::components::stage::{LoadStageFamily, Stage, StageFamily, TilingLayout};

use cubecl::std::{Swizzle, tensor::layout::Coords2d};
use cubek_std::{
    stage::{StageMemoryConfig, as_swizzle_object},
    tile::{Scope, StridedTile, Tile},
};

use crate::components::stage::reader::BiasTilingLayout;

pub struct BiasStageFamily;

impl StageFamily for BiasStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = BiasStageMemory<ES, NS>;
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct BiasStageMemory<ES: Numeric, NS: Size> {
    /// Underlying shared memory
    pub smem: SharedMemory<Vector<ES, NS>>,
    /// Swizzling of the shared memory, if any
    pub swizzle: Swizzle,
    buffer_index: u32,

    #[cube(comptime)]
    stage_size: u32,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl<ES: Numeric, NS: Size> BiasStageMemory<ES, NS> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(#[comptime] config: StageMemoryConfig) -> BiasStageMemory<ES, NS> {
        Self::new_aligned(Vector::<ES, NS>::type_size(), config)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> BiasStageMemory<ES, NS> {
        let vector_size = config.vector_size as usize;
        let swizzle = as_swizzle_object(config.swizzle);
        let swizzle_align = swizzle.repeats_after();
        let align = comptime![Ord::max(alignment, swizzle_align as usize)];
        let type_size = Vector::<ES, NS>::type_size().comptime();

        let stage_size_bytes =
            config.elements_per_stage_along_contiguous_dim() as usize * type_size;
        // Ensure all stages are aligned properly
        let stage_size = stage_size_bytes.next_multiple_of(align) / type_size / vector_size;

        let smem = SharedMemory::new_aligned(config.num_stages as usize * stage_size, align);

        BiasStageMemory::<ES, NS> {
            smem,
            swizzle,
            stage_size: stage_size as u32,
            config,
            buffer_index: 0u32,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        BiasStageMemory::<ES, NS> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: buffer_idx,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile(&self, tile: Coords2d) -> StridedTile<ES, NS> {
        BiasTilingLayout::get_tile::<ES, NS>(self, tile, self.config)
    }

    /// Get the tile at position (row, col)
    pub fn get_tile_mut(&self, tile: Coords2d) -> StridedTile<ES, NS, ReadWrite> {
        let tile = self.get_tile(tile);
        StridedTile::<ES, NS, ReadWrite> {
            container: tile.container.as_mut_unchecked(),
            start: tile.start,
            end: tile.end,
            stride: tile.stride,
            swizzle: tile.swizzle,
            layout: tile.layout,
        }
    }

    /// Return the whole stage as a slice, for reading
    pub fn as_slice(&self) -> Slice<Vector<ES, NS>> {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem
            .slice(stage_offset, stage_offset + self.stage_size as usize)
            .with_vector_size()
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self) -> SliceMut<Vector<ES, NS>> {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem
            .slice_mut(stage_offset, stage_offset + self.stage_size as usize)
            .with_vector_size()
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
impl<ES: Numeric, NS: Size> Stage<ES, NS, ReadOnly> for BiasStageMemory<ES, NS> {
    fn tile<Sc: Scope>(this: &Self, tile: Coords2d) -> Tile<ES, NS, Sc, ReadOnly> {
        Tile::new_SharedMemory(this.get_tile(tile))
    }
}

#[cube]
impl LoadStageFamily<ReadOnly> for BiasStageFamily {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        BiasStageMemory::new_aligned(alignment, config)
    }

    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, NS, T> {
        stage.with_buffer_index(buffer_index)
    }

    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>) {
        unsafe { stage.free() };
    }
}
