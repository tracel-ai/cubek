use std::marker::PhantomData;

use cubecl::{
    prelude::*,
    std::{Swizzle, tensor::layout::Coords2d},
};

use crate::{
    stage::{StageMemoryConfig, as_swizzle_object},
    tile::StridedTile,
};

use super::layout::TilingLayout;

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct StridedStageMemory<ES: Numeric, NS: Size, T: TilingLayout> {
    /// Underlying shared memory
    pub smem: SharedMemory<Vector<ES, NS>>,
    /// Swizzling of the shared memory, if any
    pub swizzle: Swizzle,
    pub(crate) buffer_index: u32,

    #[cube(comptime)]
    pub(crate) stage_size: u32,
    #[cube(comptime)]
    pub(crate) config: StageMemoryConfig,

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

        let smem = Shared::new_aligned_array(config.num_stages as usize * stage_size, align);

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

    /// Return the whole stage as a slice, for reading
    pub fn as_slice<N: Size>(&self) -> &[Vector<ES, N>] {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem[stage_offset..stage_offset + self.stage_size as usize].with_vector_size()
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut<N: Size>(&mut self) -> &mut [Vector<ES, N>] {
        let stage_offset = (self.buffer_index * self.stage_size) as usize;
        self.smem[stage_offset..stage_offset + self.stage_size as usize].with_vector_size_mut()
    }

    /// Frees the shared memory for reuse, if possible on the target runtime.
    ///
    /// # Safety
    /// *Must* be used in uniform control flow
    /// *Must not* have any dangling references to this shared memory
    pub unsafe fn free(&self) {
        unsafe { self.smem.free() };
    }
}
