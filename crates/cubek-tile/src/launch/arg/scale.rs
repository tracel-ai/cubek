//! A scale level a launch may or may not have bound, as the tile it serves.
//!
//! A scheme binds the levels it has, and a kernel written for the scheme says
//! [`mul_bound`](crate::Tile::mul_bound) once per level whether or not this launch bound it: an
//! absent level multiplies nothing and emits nothing.

use cubecl::prelude::*;

use crate::*;

/// A scale level a launch may or may not have bound, as the tile it serves: `u32` words read in
/// the level's own width, which its [`Field`](crate::Field) states and nothing here asks about.
#[cube]
pub fn scale_tile<S: Numeric>(
    level: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
    #[comptime] space: Partitioning,
) -> ComptimeOption<Tile<S>> {
    #[comptime]
    match level {
        ComptimeOption::Some(level) => {
            ComptimeOption::new_Some(level.tile_as::<S>(comptime!(space.clone())))
        }
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

    /// This level as the steps under one region of `level` read it: a stage refilled once a
    /// region with [`copy_from`](MaybeTile::copy_from). Where `level` names none the steps read
    /// the level where it lies, and this is that level.
    ///
    /// A level a scheme never bound stages nothing and stays absent, so a kernel stages its
    /// scales without asking whether it has any.
    fn staged(
        &self,
        #[comptime] level: Option<Level>,
        #[comptime] storage: StageStorage,
    ) -> ComptimeOption<Tile<Self::E>>;

    /// This level filled from `src`, where both are there; nothing where either is not.
    ///
    /// The pair is what a stage and the scales it stages are: they are bound together or not at
    /// all, so a caller says "fill" once rather than testing both.
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
