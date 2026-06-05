use crate::definition::{InterpolateOptions, TileSize, Transform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateBlueprint {
    pub tile_size: TileSize,
    pub options: InterpolateOptions,
    pub transform_width: Transform,
    pub transform_height: Transform,
    pub global: GlobalInterpolateBlueprint,
}

impl InterpolateBlueprint {
    pub fn is_flattened(&self) -> bool {
        match self.global {
            GlobalInterpolateBlueprint::GlobalMemoryBlueprint(global_memory_blueprint) => {
                global_memory_blueprint.is_flattened
            }
            GlobalInterpolateBlueprint::SharedMemoryBlueprint(_) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalInterpolateBlueprint {
    GlobalMemoryBlueprint(GlobalMemoryBlueprint),
    SharedMemoryBlueprint(SharedMemoryBlueprint),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryBlueprint {
    pub is_flattened: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryBlueprint {
    pub smem_width: usize,
    pub smem_height: usize,
    pub num_vectors: usize,
}
