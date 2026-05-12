//! Tile variants that wrap stage memory and per-partition tile collections.
//!
//! - [`memory`] / [`layout`] — the owning stage-memory wrapper (moved from
//!   `cubek-matmul`) plus the tiling-layout trait and its impls. Continues
//!   to back today's `Stage`/`LoadStageFamily` trait impls (declared in
//!   `cubek-matmul`); this module only carries the data + dispatch helpers.
//! - [`strided`] — `StridedStage<E, IO>`, the type-erased view installed as
//!   a [`TileKind::Stage`](crate::tile::TileKind) payload.
//! - [`partition`] — `PartitionTile<N, Sc, IO>`, the per-primitive
//!   collection of accumulator tiles installed as a
//!   [`TileKind::Partition`](crate::tile::TileKind) payload.

pub(crate) mod layout;
pub(crate) mod memory;
pub(crate) mod partition;
pub(crate) mod strided;

pub use layout::*;
pub use memory::*;
pub use partition::*;
pub use strided::*;
