//! A quantized operand's [`Scales`]: the scale per block, and the one factor over the whole
//! tensor that may sit above it. Two levels and no more: a coarser level is read once per leaf
//! region at its origin, so it is only correct when it covers the whole tensor.
//!
//! Which operand of a contraction a scales operand multiplies is stated by the kernel
//! ([`Scaling`]), never read off the axes: `(a ⊗ s) · b` and `a · (b ⊗ s)` are the same sum of
//! terms, and where the factor folds in is the kernel's to say.

use cubecl::prelude::*;

use crate::{Region, Tile};

/// One quantized operand's scales, windowed with [`at`](Scales::at) like the values they cover.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Scales<S: Numeric> {
    /// One scale per block: spans the block axis, omits the position inside it.
    pub block: Tile<S>,
    /// One factor over the whole tensor: spans the block level's axes, addresses none.
    pub global: ComptimeOption<Tile<S>>,
}

#[cube]
impl<S: Numeric> Scales<S> {
    pub fn block(block: Tile<S>) -> Scales<S> {
        Scales::<S> {
            block,
            global: ComptimeOption::new_None(),
        }
    }

    pub fn block_under(block: Tile<S>, global: Tile<S>) -> Scales<S> {
        Scales::<S> {
            block,
            global: ComptimeOption::new_Some(global),
        }
    }

    pub fn at(&self, region: &Region) -> Scales<S> {
        #[comptime]
        let global = match &self.global {
            ComptimeOption::Some(global) => ComptimeOption::new_Some(global.at(region)),
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        Scales::<S> {
            block: self.block.at(region),
            global,
        }
    }

    pub(crate) fn levels(&self) -> Sequence<Tile<S>> {
        let mut levels = Sequence::new();
        levels.push(self.block.clone());
        #[comptime]
        match &self.global {
            ComptimeOption::Some(global) => levels.push(global.clone()),
            ComptimeOption::None => {}
        }
        levels
    }
}

/// Which factor of the term a scales operand multiplies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleSide {
    /// Folded into the lhs value before it forms its products: once per `(row, k)`.
    Lhs,
    /// Folded into each rhs line: once per `(col, k)`.
    Rhs,
}

/// The scales of a contraction, each on the operand it multiplies. Stated by the kernel with
/// [`lhs`](Scaling::lhs) or [`rhs`](Scaling::rhs); nothing infers a side from the axes.
///
/// One side at a time for now: a contraction scaled on both takes a quantized signal against a
/// quantized weight, which is quest 4's.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Scaling<S: Numeric> {
    pub lhs: ComptimeOption<Scales<S>>,
    pub rhs: ComptimeOption<Scales<S>>,
}

#[cube]
impl<S: Numeric> Scaling<S> {
    /// `(a ⊗ s) · b`.
    pub fn lhs(scales: Scales<S>) -> Scaling<S> {
        Scaling::<S> {
            lhs: ComptimeOption::new_Some(scales),
            rhs: ComptimeOption::new_None(),
        }
    }

    /// `a · (b ⊗ s)`.
    pub fn rhs(scales: Scales<S>) -> Scaling<S> {
        Scaling::<S> {
            lhs: ComptimeOption::new_None(),
            rhs: ComptimeOption::new_Some(scales),
        }
    }

    /// [`lhs`](Scaling::lhs) or [`rhs`](Scaling::rhs) by a comptime side, for a kernel whose
    /// operand order is a comptime of its own.
    pub fn on(#[comptime] side: ScaleSide, scales: Scales<S>) -> Scaling<S> {
        match comptime!(side) {
            ScaleSide::Lhs => Scaling::<S>::lhs(scales),
            ScaleSide::Rhs => Scaling::<S>::rhs(scales),
        }
    }

    pub fn at(&self, region: &Region) -> Scaling<S> {
        #[comptime]
        let lhs = match &self.lhs {
            ComptimeOption::Some(scales) => ComptimeOption::new_Some(scales.at(region)),
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        #[comptime]
        let rhs = match &self.rhs {
            ComptimeOption::Some(scales) => ComptimeOption::new_Some(scales.at(region)),
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        Scaling::<S> { lhs, rhs }
    }

    /// The one side stated. Both, or neither, is refused here rather than deeper.
    pub(crate) fn side(&self) -> comptime_type!(ScaleSide) {
        let lhs = self.lhs.is_some();
        let rhs = self.rhs.is_some();
        comptime!(stated_side(lhs, rhs))
    }

    /// The levels of the one side stated, the block level first.
    pub(crate) fn levels(&self) -> Sequence<Tile<S>> {
        #[comptime]
        match (&self.lhs, &self.rhs) {
            (ComptimeOption::Some(scales), ComptimeOption::None)
            | (ComptimeOption::None, ComptimeOption::Some(scales)) => scales.levels(),
            _ => {
                let _ = self.side();
                Sequence::new()
            }
        }
    }
}

/// The side a [`Scaling`] states, from which of its two options is present.
fn stated_side(lhs: bool, rhs: bool) -> ScaleSide {
    match (lhs, rhs) {
        (true, false) => ScaleSide::Lhs,
        (false, true) => ScaleSide::Rhs,
        (true, true) => panic!(
            "mm_scaled: scales on both operands is a quantized signal against a quantized \
             weight, which the leaves do not fold yet; scale one side"
        ),
        (false, false) => panic!("mm_scaled: no scales on either operand; use `mm`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_side_is_the_one_option_present() {
        assert_eq!(stated_side(true, false), ScaleSide::Lhs);
        assert_eq!(stated_side(false, true), ScaleSide::Rhs);
    }

    /// A quantized signal against a quantized weight is a second contraction shape, not this
    /// one under two options.
    #[test]
    #[should_panic(expected = "scales on both operands")]
    fn both_sides_scaled_is_refused() {
        stated_side(true, true);
    }

    #[test]
    #[should_panic(expected = "no scales on either operand")]
    fn no_side_scaled_is_refused() {
        stated_side(false, false);
    }
}
