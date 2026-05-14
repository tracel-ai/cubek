//! Tile variants whose payload is a stage-shaped view or per-partition
//! collection of tiles. Just the two genuine variants:
//!
//! - [`strided`] — `StridedStage<E, IO>` (the [`TileKind::Stage`] payload)
//!   plus `SharedTile<E, NS, IO>` (the [`TileKind::SharedTile`] payload).
//! - [`partition`] — `PartitionTile<N, Sc, IO>` (the [`TileKind::Partition`]
//!   payload) plus the partition-matmul body that operates on it.
//!
//! Partition-matmul flow helpers that aren't tile variants live elsewhere:
//! [`StageEvent`](crate::tile::StageEvent) /
//! [`StageEventListener`](crate::tile::StageEventListener) /
//! [`NoEvent`](crate::tile::NoEvent) in `tile/event.rs`,
//! [`PartitionScheduler`](crate::tile::PartitionScheduler) /
//! [`PartitionBuffering`](crate::tile::PartitionBuffering) in
//! `tile/scheduler.rs`, stage-memory data types in `cubek_std::stage`.

pub(crate) mod partition;
pub(crate) mod strided;

pub use partition::*;
pub use strided::*;
