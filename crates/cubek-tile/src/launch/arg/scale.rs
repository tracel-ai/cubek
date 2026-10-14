//! A scale level that may or may not be there, windowed and staged as the values it scales.

use cubecl::prelude::*;

use crate::*;

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
