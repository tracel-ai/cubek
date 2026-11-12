//! The plane routines: every plane owns a box of queries of one head, or of keys of one key head,
//! on a matrix instruction, and stages what it walks in shared memory of its own, filled by its
//! own units and met on `sync_plane`, so no plane waits on another. Beside them, the delta
//! routine's planes each write their queries' dot products.

mod delta;
mod kernel;
mod key_gradient;
mod query_gradient;

pub use delta::*;
pub use key_gradient::*;
pub use query_gradient::*;

use cubek_std::microkernel::StageRing;

/// How a walk of the plane fragment routines lays out the shared memory it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionBackwardSharedMemory {
    /// The settings' ring as this device holds a plane's stages ([`StageRing::resolve`]).
    pub ring: StageRing,
    /// The planes of a cube, each holding its own windows.
    pub planes: usize,
}
