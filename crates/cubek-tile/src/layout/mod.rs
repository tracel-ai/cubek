//! Where an operand's axes live: [`Projection`] maps logical axes onto buffer dims,
//! [`Geometry`] holds extents and strides, [`StoragePartitioning`] how values are tiled.

mod buffer;
mod build;
mod compaction;
mod dim;
mod geometry;
mod in_kernel;
mod kernel;
mod projection;
mod row_arrangement;
mod storage_partitioning;
mod swizzle;

pub(crate) use buffer::*;
pub use build::*;
pub use compaction::*;
pub use dim::*;
pub use geometry::*;
pub(crate) use in_kernel::*;
pub(crate) use kernel::*;
pub use projection::*;
pub(crate) use row_arrangement::{LineBytes, RowArrangement};
pub use storage_partitioning::*;
pub(crate) use swizzle::ChunkSwizzle;
pub(crate) use swizzle::swizzled_line;
