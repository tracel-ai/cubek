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
    /// Not shared memory at all: the plane holds the lines in its lanes, one to a lane
    /// ([`Lanes`](crate::Lanes)). `reach` is how a value gets from the lane that loaded its line
    /// to the lane that wants it.
    Lanes {
        reach: Reach,
    },
}

/// How a value reaches the lane asking for it, out of lines the plane holds in its lanes.
///
/// The load is one coalesced transaction either way; that is not what this chooses. It chooses
/// what happens between the lane that loaded a line and the lane that wants a value out of it,
/// and the two ends are a real trade rather than a preference.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum Reach {
    /// By plane shuffle, nothing written anywhere. A read costs **one shuffle per word of the
    /// line** plus a dynamic pick, because which word a value sits in is a runtime coordinate
    /// and the shuffle's source lane is too — so every word is fetched and one is kept.
    Shuffle,
    /// Through a window of shared memory the plane owns, written once when the lines load. A
    /// read is one indexed load; the write costs a plane barrier per load and the window costs
    /// its bytes.
    Window,
}
