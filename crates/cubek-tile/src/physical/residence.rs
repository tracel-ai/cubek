//! How a shared-memory stage lays its buffer out ([`StageStorage`]).

/// How a shared-memory stage lays out its buffer: storage-tiled at the final tile (one
/// contiguous block per fragment, what a cmma transaction wants) or plain strided rows. Stated
/// by the kernel where it allocates the stage ([`Ring::smem`](crate::Ring::smem)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    Tiled,
    Strided,
}
