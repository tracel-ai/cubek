//! This instance's share of a level distributed as one flat index ([`Level::sharing`](crate::Level)):
//! the regions it touches and, for the first and last, how much of their walk below is its own.

use cubecl::prelude::*;

use crate::{ComputeScope, Coverage, CubeAxis, Level, Region, Walk, WalkExpand};

/// This instance's portion of a level distributed as one index: the regions it touches and, for the
/// first and last, how much of their walk below is its own. Counted in the level below's steps,
/// the ones its instances take *together*, however that level cuts the plane.
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

    /// Where in the `i`-th region's walk below this portion starts, and how many steps of it are
    /// its own: all of them inside the portion, part of them at either end. What the region's walk
    /// is [`range`](Walk::range)d to.
    pub fn steps(&self, i: usize) -> (usize, usize) {
        let base = (self.first + i) * self.stride;
        let from = select(base < self.start, self.start - base, 0);
        let to = select(self.end < base + self.stride, self.end - base, self.stride);
        (from, to - from)
    }
}

#[cube]
impl Walk {
    /// This instance's portion of the index this level distributes as one, counted in steps of `below`,
    /// the level each region is walked with. Two divisions rather than a length each: the portions
    /// abut, cover the work once, and differ in length by at most one.
    pub fn portion(self, #[comptime] below: Level) -> Portion {
        let instances = comptime!(
            self.level
                .shared_by()
                .expect("Walk::portion: this level distributes no grid as one; say `shared_by`")
        );
        let stride = self.region(0).over(&below).total();
        let steps = self.total() * stride;
        // A shared grid rides its scope's first dimension: the cubes' `X`, or the planes.
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
        // Through the region the portion's last step falls in. `end` is exclusive, so the step
        // before it is the one to find; an empty portion (more instances than work) touches
        // nothing.
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
