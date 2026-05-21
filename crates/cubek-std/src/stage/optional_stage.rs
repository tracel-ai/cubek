//! `Option<S>` / `ComptimeOption<S>` adapters for `Stage` / `StageFamily` /
//! `LoadStageFamily`. Lets callers thread an optional stage through the same
//! traits as a concrete one; the `None` arm produces `Tile::new_None()`.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::{
    stage::{LoadStageFamily, Stage, StageFamily, StageMemoryConfig, TilingLayout},
    tile::{Tile, TileScope},
};

#[cube]
impl<ES: Numeric, Inner: Stage<ES>> Stage<ES> for ComptimeOption<Inner> {
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc> {
        #[comptime]
        if let ComptimeOption::Some(inner) = this {
            Inner::tile::<Sc>(inner, tile)
        } else {
            Tile::new_None()
        }
    }

    fn as_stage_tile<Sc: TileScope>(this: &Self) -> Tile<ES, Sc> {
        #[comptime]
        if let ComptimeOption::Some(inner) = this {
            Inner::as_stage_tile::<Sc>(inner)
        } else {
            Tile::new_None()
        }
    }
}

impl<Inner: StageFamily> StageFamily for Option<Inner> {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = ComptimeOption<Inner::Stage<ES, NS, T>>;
}

#[cube]
impl<S: LoadStageFamily> LoadStageFamily for Option<S> {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        ComptimeOption::new_Some(S::create(alignment, config))
    }

    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        index: u32,
    ) -> Self::Stage<ES, NS, T> {
        stage.as_ref().map(|s| S::with_buffer_index(s, index))
    }

    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>) {
        #[comptime]
        if let ComptimeOption::Some(inner) = stage {
            S::free(inner)
        }
    }
}
