//! A quantized operand's [`Scales`]: the scale per block, and the one factor over the whole
//! tensor that may sit above it. Two levels and no more: a coarser level is read once per leaf
//! region at its origin, so it is only correct when it covers the whole tensor.

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
