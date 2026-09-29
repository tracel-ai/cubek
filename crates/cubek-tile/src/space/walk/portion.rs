//! This instance's share of a level distributed as one flat index.

use cubecl::prelude::*;

use crate::{ComputeScope, Coverage, CubeAxis, Level, Region, Walk, WalkExpand};

/// The regions this instance touches and how much of the first and last is its own.
/// Counted in the level below's joint steps.
#[derive(CubeType)]
pub struct Portion {
    /// The regions the portion touches, in order.
    regions: Walk,
    /// The first of them, in the level's flat index.
    first: usize,
    /// The portion's bounds on the joint step index.
    start: usize,
    end: usize,
    /// Steps of the level below per region.
    stride: usize,
}

#[cube]
impl Portion {
    /// How many regions the portion touches.
    pub fn touched(&self) -> usize {
        self.regions.total()
    }

    /// The `i`-th region the portion touches.
    pub fn region(&self, i: usize) -> Region {
        self.regions.region(i)
    }

    /// Start and length of this portion's steps within the `i`-th region's walk below.
    pub fn steps(&self, i: usize) -> (usize, usize) {
        let base = (self.first + i) * self.stride;
        let from = select(base < self.start, self.start - base, 0);
        let to = select(self.end < base + self.stride, self.end - base, self.stride);
        (from, to - from)
    }
}

#[cube]
impl Walk {
    /// This instance's portion of the level's flat index, counted in steps of `below`.
    pub fn portion(self, #[comptime] below: Level) -> Portion {
        let instances = comptime!(
            self.level
                .shared_by()
                .expect("Walk::portion: this level distributes no grid as one; say `shared_by`")
        );
        let stride = self.region(0).over(&below).total();
        let steps = self.total() * stride;
        let pos = match comptime!(self.level.coverage()) {
            Coverage::Distribute(ComputeScope::Cube) => CubeAxis::position(CubeAxis::X),
            Coverage::Distribute(ComputeScope::Plane) => {
                ComputeScope::position(ComputeScope::Plane)
            }
            Coverage::Distribute(ComputeScope::Unit) | Coverage::Walk => {
                panic!("Walk::portion: only cubes or planes share a grid as one index")
            }
        };
        let start = pos * steps / instances;
        let end = (pos + 1) * steps / instances;
        let first = start / stride;
        // `end` is exclusive; an empty portion touches nothing.
        let touched = select(start < end, (end - 1) / stride + 1 - first, 0);
        Portion {
            regions: self.range(first, touched),
            first,
            start,
            end,
            stride,
        }
    }
}
