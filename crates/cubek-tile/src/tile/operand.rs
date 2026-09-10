//! What a contraction reads as one factor of its terms: an operand's values, and the scales that
//! multiply them.
//!
//! A quantized tensor is values *and* scales, and an operand's binding names one thing, so the
//! scales are an operand of their own and folding them in is the contraction's own arithmetic.
//! Which factor they multiply is where the kernel wrote them. How deep they go is how many times
//! it said so. Nothing here states a side and nothing counts levels.
//!
//! **Nothing here multiplies anything.** `with_scale` says which level a factor carries, and the
//! multiply happens once per line, at the read, inside the leaf
//! ([`ScaledLines`](crate::ScaledLines)) — which is the only place it can happen without
//! materializing a dequantized tile. So a factor that says `with_scale` twice is multiplied twice
//! per line, not twice up front, and the verb that applies them is the contraction's
//! ([`mm_scaled`](Tile::mm_scaled) and its twins).
//!
//! A factor that carries none is [`Tile::plain`], and the leaf reads it with no arithmetic at
//! all, which is why a float kernel compiles to what it always did. A level the launch binds is
//! written with [`Tile::with_scale_arg`], once per level and named: binding none of them is
//! scaling by one.
//!
//! A level above the first is read once per leaf region at its origin, so it is only correct
//! where it covers the whole tile the level below spans.

use cubecl::prelude::*;

use crate::*;

/// A factor and the scales it carries, which are a pair and not a product: the values, and the
/// levels that will multiply them when the leaf reads a line. [`Tile::with_scale`], said once per
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

    /// This factor, carrying `level`: stated here, folded in per line by the contraction. Say it
    /// again for a level above that one.
    pub fn with_scale<S: Numeric>(&self, level: &Tile<S>) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        levels.push(level.clone());
        Scaled::<E, S> {
            values: self.clone(),
            levels,
        }
    }

    /// [`with_scale`](Tile::with_scale) for a level a launch bound, which a scheme may not have:
    /// absent, it folds nothing and emits nothing, which is scaling by one. Say it again for a
    /// level above that one, the innermost first.
    ///
    /// A bound level is stored words, read in its own width: which width is its
    /// [`Field`](crate::Field), which the launch states on it and nothing here asks about.
    pub fn with_scale_arg<S: Numeric>(
        &self,
        level: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
        #[comptime] space: Partitioning,
    ) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        #[comptime]
        match level {
            ComptimeOption::Some(level) => {
                levels.push(level.tile_as::<S>(comptime!(space.clone())))
            }
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
    /// This factor, carrying one level more.
    pub fn with_scale(&self, level: &Tile<S>) -> Scaled<E, S> {
        let mut levels = self.levels.clone();
        levels.push(level.clone());
        Scaled::<E, S> {
            values: self.values.clone(),
            levels,
        }
    }

    /// [`with_scale`](Scaled::with_scale) for one bound level more, which a scheme may not have.
    pub fn with_scale_arg(
        &self,
        level: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
        #[comptime] space: Partitioning,
    ) -> Scaled<E, S> {
        let mut levels = self.levels.clone();
        #[comptime]
        match level {
            ComptimeOption::Some(level) => {
                levels.push(level.tile_as::<S>(comptime!(space.clone())))
            }
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
