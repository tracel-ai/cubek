//! The last cube in, for a merge that is not an add.
//!
//! An atomically accumulated output serves a merge that *is* an add. One that must *read* the
//! other partials (attention's running max and its rescale) would otherwise cost a second
//! dispatch; here each cube publishes its partial, counts in on a `u32`, the last merges.
//!
//! Two things the caller owns. The counter holds zero before the launch, since arrivals are
//! counted up from there. And it counts one round, so a kernel that meets twice needs two.
//!
//! What the caller does not own is the ordering. [`Arrival::count_in`] is the fence as well as
//! the count, so the cube it answers `true` may read every partial and the cubes it answers
//! `false` may return.
//!
//! It needs a synchronization whose memory scope is the device, which WebGPU's memory model does
//! not promise. A launch reads `client.properties().features.device_memory_scope` and chooses
//! another plan (a second dispatch over the partials) when it is unset.

use cubecl::prelude::*;

use crate::algebra::comptime_only;

/// The cubes that meet on one counter.
///
/// The count is the cubes sharing the counter, not the grid: a launch whose cubes fall into
/// independent groups (one per output row, one per head) gives each group its own counter, so
/// `CUBE_COUNT` fits only when the whole grid meets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Arrival {
    cubes: u32,
}

impl Arrival {
    pub const fn new(cubes: u32) -> Self {
        Arrival { cubes }
    }
}

#[cube]
impl Arrival {
    /// Announce that this cube's partials are written, and answer whether it is the last of the
    /// cubes that share `counter`. Every unit of the cube must reach this.
    fn count_in_of(counter: &Atomic<u32>, #[comptime] arrival: Arrival) -> bool {
        let mut arrived = Shared::<u32>::new();

        // Release. Every unit of this cube has written its partial, and what they wrote is
        // visible to whichever cube counts last.
        sync_storage();
        if UNIT_POS == 0 {
            *arrived = counter.fetch_add(1u32);
        }
        // Acquire, and the broadcast the same scope carries: the count reaches every unit of the
        // cube, and what the cubes that counted before published is visible here.
        sync_storage();

        *arrived == comptime!(arrival.cubes - 1)
    }
}

/// `arrival.count_in(counter)`, the pair [`Monoid::fold`](crate::Monoid::fold) documents.
impl Arrival {
    pub fn count_in(self, counter: &Atomic<u32>) -> bool {
        Arrival::count_in_of(counter, self)
    }

    pub fn __expand_count_in_method(
        self,
        scope: &Scope,
        counter: &<Atomic<u32> as CubeType>::ExpandType,
    ) -> <bool as CubeType>::ExpandType {
        Arrival::__expand_count_in_of(scope, counter, self)
    }
}

comptime_only!(Arrival);
