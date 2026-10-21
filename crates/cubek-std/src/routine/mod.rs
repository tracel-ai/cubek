//! A kernel written around one microkernel: the [`Routine`] trait, the [`Grid`] it launches on,
//! and the [`LaunchSettings`] that grid is checked as.
//!
//! A routine is the one place a kernel's hardware assumptions are stated. It takes the problem
//! and the [`Hardware`](crate::Hardware), and returns a blueprint of compile-time decisions and
//! the grid to run on, or a [`Refusal`](cubek_tile::launch::Refusal) saying why this device
//! cannot run the kernel. Anything it decides is explicit and testable; anything it cannot
//! decide, it declines.
//!
//! # What goes where
//!
//! The blueprint holds decisions distilled from the hardware and the problem, never the raw
//! values: algorithm variants, safety strategies, shared-memory sizing, loop bounds to unroll.
//! `cube_dim` and `cube_count` are not in it (they are already in the compilation key), nor are
//! vector sizes (carried by the tensor types), hardware properties (readable in the kernel) or
//! problem shapes (runtime arguments).
//!
//! # `prepare` is pure
//!
//! No device reads, no allocation, deterministic. Graph capture replays a recorded launch: if
//! `prepare` could observe device state or vary between calls, a captured step would drift from
//! the one that was recorded. Everything it needs is host data.
//!
//! # The grid is stated twice and checked
//!
//! The division counts the grid; the space states it, on whichever axes the cut puts it; the
//! derivation asserts the two agree and holds every candidate to the device's caps
//! ([`LaunchSettings::validate`]). A grid this device does not launch declines that candidate.
//!
//! # A refusal is for the selector
//!
//! Autotune drops a candidate that returns `Err`, so at least one routine of a family must be
//! launchable everywhere. A capability query turns `Err` into `false`. Otherwise the launch
//! expects on it, and the refusal's message is the whole deliverable.

mod base;
mod grid;
mod launch_settings;

pub use base::*;
pub use grid::*;
pub use launch_settings::*;
