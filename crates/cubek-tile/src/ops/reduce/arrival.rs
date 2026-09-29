//! The last cube in, for a cross-cube merge that is not an add.
//! The counter must be zero before launch and serves one round. Requires a device memory
//! scope (`features.device_memory_scope`).

use cubecl::prelude::*;

use crate::algebra::comptime_only;

/// The cubes that meet on one counter (those sharing it, not the grid).
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

        // Release: this cube's partials become visible to the last cube.
        sync_storage();
        if UNIT_POS == 0 {
            *arrived = counter.fetch_add(1u32);
        }
        // Acquire: earlier cubes' partials become visible, and the count reaches every unit.
        sync_storage();

        *arrived == comptime!(arrival.cubes - 1)
    }
}

/// `arrival.count_in(counter)`, the pair [`Monoid::combine`](crate::Monoid::combine) documents.
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
