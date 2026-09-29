//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::Axis;

/// Where a stage lives and how it lays its cells out, stated at
/// [`Stages::smem`](crate::Stages::smem).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Shared memory grouped into `block`-sized tiles, one edge per axis of the operation.
    Tiled {
        block: Vec<(Axis, usize)>,
        chunks: RowChunks,
    },
    Strided,
    /// The plane's units, one line per unit.
    Lines {
        read: UnitRead,
    },
}

/// How a [`Tiled`](StageStorage::Tiled) block lays each row down, in 16-byte chunks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum RowChunks {
    /// Each row one contiguous run, in order; readable by `cmma` and TMA.
    InOrder,
    /// Each row's chunks XOR-permuted by a row key; only per-unit-addressed readers can read it.
    Swizzled,
    /// Each row in order followed by one unused chunk, so rows start on distinct banks.
    Padded,
}

impl RowChunks {
    /// Bytes one chunk holds: the widest read one unit issues.
    pub const CHUNK_BYTES: usize = 16;
}

/// How a value reaches the unit asking for it, out of the lines the plane holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum UnitRead {
    /// By plane shuffle: one shuffle per word of the line plus a dynamic pick.
    Shuffle,
    /// Through a plane-owned shared-memory window written once per load. Serialized as `Window`.
    #[serde(rename = "Window")]
    PlaneShared,
}
