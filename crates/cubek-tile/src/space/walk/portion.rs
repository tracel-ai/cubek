//! This instance's run of a level dealt as one index ([`Level::shared_by`](crate::Level)).

use cubecl::prelude::*;

use crate::{Level, Region, Walk, WalkExpand, hardware_pos};

/// This instance's run of a level dealt as one index ([`Level::shared_by`]): the regions it
/// touches and, for the first and last, how much of their walk below is its own. Counted in the
/// level below's steps, the ones its instances take *together*, however that level cuts the plane.
#[derive(CubeType)]
pub struct Run {
    /// The regions the run touches, in order.
    regions: Walk,
    /// The first of them, in the level's flat index.
    first: usize,
    /// The run's bounds on the joint step index.
    start: usize,
    end: usize,
    /// Steps of the level below per region.
    stride: usize,
}

#[cube]
impl Run {
    /// How many regions the run touches.
    pub fn touched(&self) -> usize {
        self.regions.total()
    }

    /// The `i`-th region the run touches.
    pub fn region(&self, i: usize) -> Region {
        self.regions.region(i)
    }

    /// Where in the `i`-th region's walk below this run starts, and how many steps of it are the
    /// run's own: all of them inside the run, part of them at either end. What the region's walk
    /// is [`window`](Walk::window)ed to.
    pub fn steps(&self, i: usize) -> (usize, usize) {
        let base = (self.first + i) * self.stride;
        let from = select(base < self.start, self.start - base, 0);
        let to = select(self.end < base + self.stride, self.end - base, self.stride);
        (from, to - from)
    }
}

#[cube]
impl Walk {
    /// This instance's run of the index this level deals as one, counted in steps of `below`,
    /// the level each region is walked with. Two divisions rather than a length each: the runs
    /// abut, cover the work once, and differ in length by at most one.
    pub fn run(self, #[comptime] below: Level) -> Run {
        let work = comptime!(
            self.level
                .work()
                .cloned()
                .expect("Walk::run: this level deals no work as one; say `shared_by`")
        );
        let instances = comptime!(work.instances());
        let stride = self.region(0).over(&below).total();
        let steps = self.total() * stride;
        let pos = hardware_pos(comptime!(work.scope()));
        let start = pos * steps / instances;
        let end = (pos + 1) * steps / instances;
        let first = start / stride;
        // Through the region the run's last step falls in. `end` is exclusive, so the step before
        // it is the one to find; an empty run (more instances than work) touches nothing.
        let touched = select(start < end, (end - 1) / stride + 1 - first, 0);
        Run {
            regions: self.window(first, touched),
            first,
            start,
            end,
            stride,
        }
    }
}
