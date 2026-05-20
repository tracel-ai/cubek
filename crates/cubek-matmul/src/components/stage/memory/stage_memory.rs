//! `Stage` / `LoadStageFamily` trait impls for the relocated
//! [`StridedStageMemory`]. The struct itself and its inherent methods live in
//! `cubek_std::tile::variants::stage`; this file keeps the
//! `cubek_matmul::components::stage::memory::stage_memory` path working for
//! callers, declares [`StridedStageFamily`] (the family wrapper that requires
//! the local `StageFamily` trait), and binds the local
//! `Stage` / `LoadStageFamily` traits to the moved type. Deleted in PR 6 once
//! the trait stack is replaced.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::{
    stage::StageMemoryConfig,
    tile::{SharedTile, Tile, TileScope},
};

pub use cubek_std::tile::StridedStageMemory;

use crate::components::stage::{LoadStageFamily, Stage, StageFamily, TilingLayout};

pub struct StridedStageFamily;

impl StageFamily for StridedStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = StridedStageMemory<ES, NS, T>;
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES> for StridedStageMemory<ES, NS, T> {
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc> {
        let strided_tile = this.get_tile(tile);
        Tile::new_SharedMemory(SharedTile::wrap::<NS>(strided_tile))
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
