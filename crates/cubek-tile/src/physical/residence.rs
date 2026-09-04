//! How a shared-memory stage lays its buffer out ([`StageStorage`]).

use crate::Axis;

/// How a shared-memory stage lays out its buffer: storage-tiled at a stated block (one contiguous
/// block per fragment, what a cmma transaction wants) or plain strided rows. Stated by the kernel
/// where it allocates the stage ([`Ring::smem`](crate::Ring::smem)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Grouped into `block`-sized tiles, the fragment the instruction reads: one edge per axis of
    /// the operation, of which an operand takes its own.
    Tiled { block: Vec<(Axis, usize)> },
    Strided,
}

impl StageStorage {
    /// [`Tiled`](StageStorage::Tiled) at `block`, one edge per axis of the operation.
    pub fn tiled(block: &[(Axis, usize)]) -> Self {
        StageStorage::Tiled {
            block: block.to_vec(),
        }
    }

    /// [`tiled`](StageStorage::tiled) at the leaf `space` is tiled to: what a kernel that does
    /// not state levels itself reads off the space it was handed.
    pub fn tiled_at_leaf(space: &crate::Space) -> Self {
        let leaf = space.final_space();
        StageStorage::tiled(
            &leaf
                .axes()
                .map(|axis| (axis, leaf.extent(axis)))
                .collect::<Vec<_>>(),
        )
    }
}
