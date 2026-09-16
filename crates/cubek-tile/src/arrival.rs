//! The last cube in, for a merge that is not an add.
//!
//! [`Tile::of_atomic_accumulate`](crate::Tile) serves every merge that *is* an add: a cube adds
//! its slice into the cell and never learns that the others exist. A merge that has to *read*
//! what the others wrote cannot be spelled that way — attention's running max, and the rescale
//! that follows from it, is the case — and closing it with a second dispatch costs a launch and
//! a round trip through memory for every partial. This is the other way, and it costs one `u32`:
//! every cube publishes its partial where the others can see it, announces itself on a counter,
//! and the cube that arrives last merges them all in place.
//!
//! Two things the caller owns. The counter holds zero before the launch, since arrivals are
//! counted up from there. And it counts one round, so a kernel that meets twice needs two.
//!
//! What the caller does not own is the ordering. [`last_cube_in`] is the fence as well as the
//! count, so the cube it answers `true` may read every partial and the cubes it answers `false`
//! may return.
//!
//! It asks the device for the only thing here that is not universal: a synchronization whose
//! memory scope is the device, which WebGPU's memory model does not promise. A launch states
//! that the way it states any other device fact, by reading
//! `client.properties().features.device_memory_scope` and choosing another plan when it is
//! unset — a second dispatch over the partials is what that other plan looks like.

use cubecl::prelude::*;

/// Announce that this cube's partials are written, and answer whether it is the last of the
/// `cubes` that share `counter`.
///
/// The count is the cubes sharing the counter, not the grid: a launch whose cubes fall into
/// independent groups — one per output row, one per head — gives each group a counter of its
/// own and merges them at the same time, so `CUBE_COUNT` is the answer only when the whole grid
/// meets. Every unit of the cube reaches this, as for any synchronization: the count is taken
/// once and read by all of them.
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
