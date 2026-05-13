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

pub(crate) mod event;
pub(crate) mod layout;
pub(crate) mod matmul;
pub(crate) mod memory;
pub(crate) mod partition;
pub(crate) mod scheduler;
pub(crate) mod strided;
pub(crate) mod tile_matmul;

pub use event::*;
pub use layout::*;
pub use matmul::*;
pub use memory::*;
pub use partition::*;
pub use scheduler::*;
pub use strided::*;
pub use tile_matmul::*;
