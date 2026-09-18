//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::Axis;

/// Where a stage lives and how it lays its cells out: shared memory, storage-tiled at a stated
/// block (one contiguous block per fragment, what a cmma transaction wants) or in plain strided
/// rows; or the plane's own lanes, which is no buffer at all. Stated by the kernel where it
/// allocates the stage ([`Ring::smem`](crate::Ring::smem)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Grouped into `block`-sized tiles, the fragment the instruction reads: one edge per axis of
    /// the operation, of which an operand takes its own.
    Tiled {
        block: Vec<(Axis, usize)>,
    },
    Strided,
    /// Not shared memory at all: the plane holds the lines in its lanes, one to a lane, and
    /// shares them by shuffle ([`Lanes`](crate::Lanes)).
    Lanes,
}
