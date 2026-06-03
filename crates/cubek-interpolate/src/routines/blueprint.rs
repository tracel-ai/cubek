use crate::definition::{InterpolateOptions, TileSize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateBlueprint {
    pub tile_size: TileSize,
    pub options: InterpolateOptions,
    pub global: GlobalInterpolateBlueprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalInterpolateBlueprint {
    GlobalMemoryBlueprint(GlobalMemoryBlueprint),
    SharedMemoryBlueprint(SharedMemoryBlueprint),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryBlueprint {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryBlueprint {
    pub smem_width: usize,
    pub smem_height: usize,
    pub num_vectors: usize,
}
