//! `Stage` / `StageFamily` / `LoadStageFamily` traits.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::{
    stage::{StageMemoryConfig, TilingLayout},
    tile::{Tile, TileScope},
};

/// A stage that can be sliced into tiles or wrapped whole as a tile.
#[cube]
pub trait Stage<ES: Numeric, IO: SliceVisibility = ReadOnly>:
    CubeType + Clone + Send + Sync + 'static
{
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, IO>;
    /// Wrap the whole stage as a [`TileKind::Stage`](crate::tile::TileKind::Stage)
    /// tile. Only meaningful for read-only stages consumed by the partition
    /// matmul; other impls return [`Tile::new_None()`].
    fn as_stage_tile<Sc: TileScope>(this: &Self) -> Tile<ES, Sc, IO>;
}

pub trait StageFamily<IO: SliceVisibility = ReadOnly>: Send + Sync + 'static {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout>: Stage<ES, IO>;
}

/// A `StageFamily` that can be allocated as a loader target.
#[cube]
pub trait LoadStageFamily<IO: SliceVisibility = ReadOnly>: StageFamily {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T>;
    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, NS, T>;
    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>);
}
