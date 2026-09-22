//! The last cube in, for a merge that is not an add.
//!
//! [`Tile::of_atomic_accumulate`](crate::Tile) serves a merge that *is* an add. One that must
//! *read* the other partials (attention's running max and its rescale) would otherwise cost a
//! second dispatch; here each cube publishes its partial, counts in on a `u32`, the last merges.
//!
//! Two things the caller owns. The counter holds zero before the launch, since arrivals are
//! counted up from there. And it counts one round, so a kernel that meets twice needs two.
//!
//! What the caller does not own is the ordering. [`last_cube_in`] is the fence as well as the
//! count, so the cube it answers `true` may read every partial and the cubes it answers `false`
//! may return.
//!
//! It needs a synchronization whose memory scope is the device, which WebGPU's memory model does
//! not promise. A launch reads `client.properties().features.device_memory_scope` and chooses
//! another plan (a second dispatch over the partials) when it is unset.

use cubecl::prelude::*;

/// Announce that this cube's partials are written, and answer whether it is the last of the
/// `cubes` that share `counter`.
///
/// The count is the cubes sharing the counter, not the grid: a launch whose cubes fall into
/// independent groups (one per output row, one per head) gives each group its own counter, so
/// `CUBE_COUNT` fits only when the whole grid meets. Every unit of the cube must reach this.
#[cube]
pub fn last_cube_in(counter: &Atomic<u32>, #[comptime] cubes: u32) -> bool {
    let mut arrived = Shared::<u32>::new();

    // Release. Every unit of this cube has written its partial, and what they wrote is visible
    // to whichever cube counts last.
    sync_storage();
    if UNIT_POS == 0 {
        *arrived = counter.fetch_add(1u32);
    }
    // Acquire, and the broadcast the same scope carries: the count reaches every unit of the
    // cube, and what the cubes that counted before published is visible here.
    sync_storage();

    *arrived == comptime!(cubes - 1)
}
