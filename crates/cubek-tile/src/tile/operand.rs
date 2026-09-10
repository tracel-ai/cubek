//! What a contraction reads as one factor of its terms: an operand's values, and the scales that
//! multiply them.
//!
//! A quantized tensor is values *and* scales, and an operand's binding names one thing, so the
//! scales are an operand of their own and folding them in is the contraction's own arithmetic.
//! Which factor they multiply is where the kernel wrote them. How deep they go is how many times
//! it said so. Nothing here states a side and nothing counts levels.
//!
//! A factor that carries none is [`Tile::plain`], and the leaf reads it with no arithmetic at
//! all, which is why a float kernel compiles to what it always did.
//!
//! A level above the first is read once per leaf region at its origin, so it is only correct
//! where it covers the whole tile the level below spans.

use cubecl::prelude::*;

use crate::*;

/// A factor and the scales it carries: [`Tile::scaled`], said once per level.
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
        let none = ComptimeOption::<Tile<E>>::new_None();
        self.maybe_scaled(&none)
    }

    /// This factor, scaled by `level`. Say it again for a level above that one.
    pub fn scaled<S: Numeric>(&self, level: &Tile<S>) -> Scaled<E, S> {
        let mut levels = Sequence::new();
        levels.push(level.clone());
        Scaled::<E, S> {
            values: self.clone(),
            levels,
        }
    }

    /// [`scaled`](Tile::scaled) by a level this scheme may not have: absent, it folds nothing and
    /// emits nothing, so the kernel writes the same line either way.
    pub fn maybe_scaled<S: Numeric>(&self, level: &ComptimeOption<Tile<S>>) -> Scaled<E, S> {
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
    /// This factor, scaled by one level more.
    pub fn scaled(&self, level: &Tile<S>) -> Scaled<E, S> {
        let mut levels = self.levels.clone();
        levels.push(level.clone());
        Scaled::<E, S> {
            values: self.values.clone(),
            levels,
        }
    }

    /// [`scaled`](Scaled::scaled) by a level this scheme may not have.
    pub fn maybe_scaled(&self, level: &ComptimeOption<Tile<S>>) -> Scaled<E, S> {
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
