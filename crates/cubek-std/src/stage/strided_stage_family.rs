//! `StridedStageFamily` — the `LoadStageFamily` wrapper around
//! [`StridedStageMemory`], plus the `Stage` impls for that memory type.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::{
    stage::{
        LoadStageFamily, Stage, StageFamily, StageMemoryConfig, StridedStageMemory, TilingLayout,
    },
    tile::{SharedTile, StageTile, Tile, TileScope},
};

pub struct StridedStageFamily;

impl StageFamily for StridedStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = StridedStageMemory<ES, NS, T>;
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES> for StridedStageMemory<ES, NS, T> {
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc> {
        let strided_tile = this.get_tile(tile);
        Tile::new_SharedTile(SharedTile::wrap::<NS>(strided_tile))
    }

    fn as_stage_tile<Sc: TileScope>(this: &Self) -> Tile<ES, Sc> {
        Tile::new_Stage(StageTile::wrap::<NS, T>(this))
    }
}

#[cube]
impl LoadStageFamily for StridedStageFamily {
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
