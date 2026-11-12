//! How a walk of the plane fragment routines lays out its shared memory.

use cubek_std::microkernel::StageRing;

/// How a walk of the plane fragment routines lays out the shared memory it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionBackwardSharedMemory {
    /// The settings' ring as this device holds a plane's stages ([`StageRing::resolve`]).
    pub ring: StageRing,
    /// The planes of a cube, each holding its own windows.
    pub planes: usize,
}
