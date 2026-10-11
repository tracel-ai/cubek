//! A scale level a launch may or may not have bound, as the tile it serves.

use cubecl::prelude::*;

use crate::*;

/// A scale level a launch may or may not have bound, as the tile it serves.
#[cube]
pub fn scale_tile<S: Numeric>(
    level: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
    partitioning: &Partitioning,
) -> ComptimeOption<Tile<S>> {
    #[comptime]
    match level {
        ComptimeOption::Some(level) => ComptimeOption::new_Some(level.tile_as::<S>(partitioning)),
        ComptimeOption::None => ComptimeOption::new_None(),
    }
}

/// A scale level windowed the way the factor it multiplies is, or nothing where none was bound.
#[cube]
pub trait MaybeTile: CubeType {
    /// The element the level is served at.
    type E: Numeric;

    /// This level at `region`, descending it as [`Tile::at`] descends a factor.
    fn at(&self, region: &Region) -> ComptimeOption<Tile<Self::E>>;

    /// This level staged per region of `level`, or as it lies where `level` is `None`.
    fn staged(
        &self,
        #[comptime] level: Option<Level>,
        #[comptime] storage: StageStorage,
    ) -> ComptimeOption<Tile<Self::E>>;

    /// This level filled from `src`, where both are there; nothing where either is not.
    fn copy_from(&mut self, src: &ComptimeOption<Tile<Self::E>>);
}

#[cube]
impl<E: Numeric> MaybeTile for ComptimeOption<Tile<E>> {
    type E = E;

    fn at(&self, region: &Region) -> ComptimeOption<Tile<E>> {
        #[comptime]
        match self {
            ComptimeOption::Some(level) => ComptimeOption::new_Some(level.at(region)),
            ComptimeOption::None => ComptimeOption::new_None(),
        }
    }

    fn staged(
        &self,
        #[comptime] level: Option<Level>,
        #[comptime] storage: StageStorage,
    ) -> ComptimeOption<Tile<E>> {
        match comptime!(level) {
            None => self.clone(),
            Some(level) =>
            {
                #[comptime]
                match self {
                    ComptimeOption::Some(scales) => ComptimeOption::new_Some(
                        scales.stage(comptime!(level.clone()), comptime!(storage.clone())),
                    ),
                    ComptimeOption::None => ComptimeOption::new_None(),
                }
            }
        }
    }

    fn copy_from(&mut self, src: &ComptimeOption<Tile<E>>) {
        #[comptime]
        if let (ComptimeOption::Some(stage), ComptimeOption::Some(src)) = (self, src) {
            stage.copy_from(src);
        }
    }
}
