//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::Axis;

/// Where a stage lives and how it lays its cells out: shared memory, storage-tiled at a stated
/// block (one contiguous block per fragment, what a cmma transaction wants) or plain strided rows;
/// or the plane's units, no buffer at all. Stated at [`Stages::smem`](crate::Stages::smem).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Grouped into `block`-sized tiles, the fragment the instruction reads: one edge per axis of
    /// the operation, of which an operand takes its own. `chunks` is how each row of a block lays
    /// its bytes down.
    Tiled {
        block: Vec<(Axis, usize)>,
        chunks: RowChunks,
    },
    Strided,
    /// Not shared memory at all: the plane holds the lines in its units, one to a unit
    /// ([`Lines`](crate::Lines)). `read` is how a value gets from the unit that loaded its line
    /// to the unit that wants it.
    Lines {
        read: UnitRead,
    },
}

/// How a [`Tiled`](StageStorage::Tiled) block lays each of its rows down, in chunks of 16 bytes:
/// the widest read one unit issues.
///
/// A fragment load has every unit of a plane read its own row of the block at once. A row shorter
/// than the shared memory's bank period starts on the same banks as the rows one period of rows
/// away, so rows kept in order serialize those reads: at a 64-byte row, four units of every eight
/// share their banks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum RowChunks {
    /// Each row one contiguous run, in order: what a fragment API that addresses the block's rows
    /// by a pointer and a stride reads (`cmma`, and a TMA box).
    InOrder,
    /// Each row's chunks permuted by an XOR with a key read off the row, so the rows a fragment
    /// load reads together start on distinct banks. Only a reader that computes each unit's
    /// address reads it: the stage's views, and the manual mma transport.
    Swizzled,
    /// Each row in order and followed by one chunk nothing reads, so the rows a fragment load reads
    /// together start on distinct banks at the cost of that chunk a row: only the stage's strides
    /// and its size change. A row is still a run a stride from the next, so a fragment API reads
    /// it too.
    Padded,
}

impl RowChunks {
    /// Bytes one chunk holds: the widest read one unit issues (`ds_load_b128`, `ld.shared.v4`), so
    /// a swizzled chunk is still read whole, and what a padded row is followed by.
    pub const CHUNK_BYTES: usize = 16;
}

/// How a value reaches the unit asking for it, out of the lines the plane holds in its units.
///
/// The load is one coalesced transaction either way; that is not what this chooses. It chooses
/// what happens between the unit that loaded a line and the unit that wants a value out of it,
/// and the two ends are a real trade rather than a preference.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum UnitRead {
    /// By plane shuffle, nothing written anywhere. A read costs **one shuffle per word of the
    /// line** plus a dynamic pick, because which word a value sits in is a runtime coordinate
    /// and the shuffle's source unit is too — so every word is fetched and one is kept.
    Shuffle,
    /// Through a window of shared memory the plane owns, written once when the lines load. A
    /// read is one indexed load; the write costs a plane barrier per load and the window costs
    /// its bytes.
    ///
    /// Spelled `Window` on disk, which is what a persisted autotune key was measured under.
    #[serde(rename = "Window")]
    PlaneShared,
}
