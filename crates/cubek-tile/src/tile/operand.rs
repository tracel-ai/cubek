//! What a contraction reads as one factor of its terms: an operand's values, and the scales that
//! multiply them.
//!
//! A quantized tensor is values *and* scales, and an operand's binding names one thing, so the
//! scales are an operand of their own and folding them in is the contraction's own arithmetic.
//! Which factor they multiply is where the kernel wrote them. How deep they go is how many times
//! it said so. Nothing here states a side and nothing counts levels.
//!
//! **Nothing here multiplies anything.** `scaled` says which level a factor carries, and the
//! multiply happens once per line, at the read, inside the leaf
//! ([`ScaledLines`](crate::ScaledLines)) — which is the only place it can happen without
//! materializing a dequantized tile. So a factor that says `scaled` twice is multiplied twice
//! per line, not twice up front, and the verb that applies them is the contraction's
//! ([`mm_scaled`](Tile::mm_scaled) and its twins).
//!
//! A factor that carries none is [`Tile::plain`], and the leaf reads it with no arithmetic at
//! all, which is why a float kernel compiles to what it always did. A level the launch binds is
//! written with [`Tile::scaled`], once per level and named: binding none of them is
//! scaling by one.
//!
//! A level above the first is read once per leaf region at its origin, so it is only correct
//! where it covers the whole tile the level below spans.

use cubecl::prelude::*;

use crate::*;

/// A factor and the scales it carries, which are a pair and not a product: the values, and the
/// levels that will multiply them when the leaf reads a line. [`Tile::scaled`], said once per
/// level.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Scaled<E: Numeric, S: Numeric> {
    values: Tile<E>,
    levels: Sequence<Tile<S>>,
}

#[cube]
impl<E: Numeric> Tile<E> {
    /// This tile as a factor carrying no scales: what a float kernel contracts, and what
    /// [`mm`](Tile::mm) hands the leaf for each of its operands.
    pub fn plain(&self) -> Scaled<E, E> {
        self.unscaled::<E>()
    }

    /// [`plain`](Tile::plain) at a scale element of its own, for a factor whose levels are about
    /// to be pushed onto it.
    fn unscaled<S: Numeric>(&self) -> Scaled<E, S> {
        Scaled::<E, S> {
            values: self.clone(),
            levels: Sequence::new(),
        }
    }

    /// This factor, carrying `level`, which a scheme may not have: absent, it folds
    /// nothing and emits nothing, which is scaling by one.
    ///
    /// A scale that does not cover everything an accumulator sums has to ride its factor like
    /// this, because the running sum already holds terms it does not apply to. One that does
    /// cover everything belongs on the accumulator instead ([`Tile::scale`]), where it costs one
    /// multiply per cell rather than one per value read.
    pub fn scaled<S: Numeric>(&self, level: &ComptimeOption<Tile<S>>) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        #[comptime]
        match level {
            ComptimeOption::Some(level) => levels.push(level.clone()),
            ComptimeOption::None => {}
        }
        Scaled::<E, S> {
            values: self.clone(),
            levels,
        }
    }
}

#[cube]
impl<E: Numeric, S: Numeric> Scaled<E, S> {
    /// This factor, carrying one level more, which a scheme may not have.
    pub fn scaled(&self, level: &ComptimeOption<Tile<S>>) -> Scaled<E, S> {
        let mut levels = self.levels.clone();
        #[comptime]
        match level {
            ComptimeOption::Some(level) => levels.push(level.clone()),
            ComptimeOption::None => {}
        }
        Scaled::<E, S> {
            values: self.values.clone(),
            levels,
        }
    }

    /// This factor windowed to `region`: the values and every level at once, each resolving the
    /// same position to its own granularity.
    pub fn at(&self, region: &Region) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        #[unroll]
        for i in 0..self.levels.len() {
            levels.push(self.levels.index(i).at(region));
        }
        Scaled::<E, S> {
            values: self.values.at(region),
            levels,
        }
    }

    /// The values this factor contributes.
    pub(crate) fn values(&self) -> Tile<E> {
        self.values.clone()
    }

    /// Every scale that multiplies them, innermost first. Empty is a factor carrying none.
    pub(crate) fn levels(&self) -> Sequence<Tile<S>> {
        self.levels.clone()
    }
}

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
}
