use cubecl::prelude::*;
use cubek_std::{stage::StageMemoryConfig, stage::as_swizzle_object, tile::StridedTile};
use std::marker::PhantomData;

use crate::components::{
    global::{GlobalReaderConfig, PlaneFlowPartition, read::StageBuffer},
    stage::{LoadStageFamily, Stage, StageFamily, TilingLayout},
    tile_matmul::{Scope, Tile},
};
use cubecl::std::{Swizzle, tensor::layout::Coords2d};

pub struct StridedStageFamily;

impl StageFamily for StridedStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = StridedStageMemory<ES, NS, T>;
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct StridedStageMemory<ES: Numeric, NS: Size, T: TilingLayout> {
    /// Underlying shared memory
    pub smem: SharedMemory<Vector<ES, NS>>,
    /// Swizzling of the shared memory, if any
    pub swizzle: Swizzle,
    buffer_index: u32,

    #[cube(comptime)]
    stage_size: u32,
    #[cube(comptime)]
    config: StageMemoryConfig,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> StridedStageMemory<ES, NS, T> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(#[comptime] config: StageMemoryConfig) -> StridedStageMemory<ES, NS, T> {
        Self::new_aligned(Vector::<ES, NS>::type_size(), config)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStageMemory<ES, NS, T> {
        let vector_size = config.vector_size as usize;
        let swizzle = as_swizzle_object(config.swizzle);
        let swizzle_align = swizzle.repeats_after();
        let align = comptime![Ord::max(alignment, swizzle_align as usize)];
        let type_size = Vector::<ES, NS>::type_size().comptime();

        let stage_size_bytes = config.elements_per_stage() as usize * type_size;
        // Ensure all stages are aligned properly
        let stage_size = stage_size_bytes.next_multiple_of(align) / type_size / vector_size;

        let smem = SharedMemory::new_aligned(config.num_stages as usize * stage_size, align);

        StridedStageMemory::<ES, NS, T> {
            smem,
            swizzle,
            stage_size: stage_size as u32,
            config,
            buffer_index: 0u32,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        StridedStageMemory::<ES, NS, T> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: buffer_idx,
            _phantom: PhantomData::<T>,
        }
    }

    /// Return the same stage but with a different tiling layout.
    /// Allows comptime switching tiling.
    pub fn with_layout<TNew: TilingLayout>(&self) -> StridedStageMemory<ES, NS, TNew> {
        StridedStageMemory::<ES, NS, TNew> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: self.buffer_index,
            _phantom: PhantomData::<TNew>,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile(&self, tile: Coords2d) -> StridedTile<ES, NS> {
        T::get_tile::<ES, NS>(self, tile, self.config)
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
    pub fn as_slice<N: Size>(&self) -> Slice<Vector<ES, N>> {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem
            .slice(stage_offset, stage_offset + self.stage_size as usize)
            .with_vector_size()
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut<N: Size>(&mut self) -> SliceMut<Vector<ES, N>> {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem
            .slice_mut(stage_offset, stage_offset + self.stage_size as usize)
            .with_vector_size()
    }

    /// Zero out the shared memory
    /// Available for matmul only
    pub fn clear_all(&mut self, #[comptime] config: GlobalReaderConfig) {
        // TODO: this assumes the stage was created with new
        let smem_length = comptime!(self.config.num_stages * self.stage_size);

        let unit_count = config.loading_units_count();
        let num_writes_per_unit = smem_length.div_ceil(unit_count);

        let unit_base_position = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        for i in 0..num_writes_per_unit {
            let offset = unit_base_position + i * unit_count;

            #[allow(clippy::collapsible_else_if)]
            if smem_length % unit_count == 0 {
                self.smem[offset as usize] = Vector::zeroed();
            } else {
                if offset < smem_length {
                    self.smem[offset as usize] = Vector::zeroed();
                }
            }
        }
    }

    /// Zero out the shared memory for only one stage
    /// Available for matmul only
    pub fn clear_stage(
        &mut self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut this = self.with_buffer_index(stage_buffer.to_index());

        let unit_count = config.loading_units_count();
        let num_writes_per_unit = this.stage_size.comptime().div_ceil(unit_count);

        let unit_base_position = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        let mut stage = this.as_slice_mut::<NS>();

        for i in 0..num_writes_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            #[allow(clippy::collapsible_else_if)]
            if this.stage_size.comptime().is_multiple_of(unit_count) {
                stage[unit_position as usize] = Vector::zeroed();
            } else {
                if unit_position < this.stage_size {
                    stage[unit_position as usize] = Vector::zeroed();
                }
            }
        }
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
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES, NS, ReadOnly>
    for StridedStageMemory<ES, NS, T>
{
    fn tile<Sc: Scope>(this: &Self, tile: Coords2d) -> Tile<ES, NS, Sc, ReadOnly> {
        let strided_tile = this.get_tile(tile);
        Tile::new_SharedMemory(strided_tile)
    }
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES, NS, ReadWrite>
    for StridedStageMemory<ES, NS, T>
{
    fn tile<Sc: Scope>(this: &Self, tile: Coords2d) -> Tile<ES, NS, Sc, ReadWrite> {
        let strided_tile = this.get_tile_mut(tile);
        Tile::new_SharedMemory(strided_tile)
    }
}

#[cube]
impl LoadStageFamily<ReadOnly> for StridedStageFamily {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        StridedStageMemory::new_aligned(alignment, config)
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
