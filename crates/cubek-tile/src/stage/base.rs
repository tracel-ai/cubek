//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::{Axis, Space};

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

impl StageStorage {
    /// The storage-tiling nesting a stage over `space` gets, coarse to fine; empty is row-major.
    pub(crate) fn nesting(&self, space: &Space) -> Vec<Space> {
        match self {
            StageStorage::Lines { .. } => {
                panic!("StageStorage::Lines: the plane's units are not shared memory")
            }
            StageStorage::Tiled { block, .. } => {
                let nested = Space::new(
                    &space
                        .axes()
                        .map(|axis| {
                            let edge = block
                                .iter()
                                .find(|&&(a, _)| a == axis)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "StageStorage::Tiled: the block states no edge for {axis:?}"
                                    )
                                })
                                .1;
                            (axis, edge)
                        })
                        .collect::<Vec<_>>(),
                );
                if &nested == space {
                    return Vec::new();
                }
                vec![nested]
            }
            StageStorage::Strided => Vec::new(),
        }
    }
}
