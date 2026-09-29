//! Where an operand's axes live: [`Projection`](crate::Projection) maps logical axes onto buffer dims,
//! [`Geometry`](crate::Geometry) holds extents and strides, [`StoragePartitioning`] how values are tiled.

pub(crate) mod buffer;
pub(crate) mod build;
pub(crate) mod compaction;
pub(crate) mod dim;
pub(crate) mod geometry;
pub(crate) mod in_kernel;
pub(crate) mod kernel;
pub(crate) mod projection;
pub(crate) mod row_arrangement;
pub(crate) mod storage_partitioning;
pub(crate) mod swizzle;
pub(crate) mod vector_tile;

pub(crate) use buffer::*;
pub(crate) use dim::*;
pub(crate) use in_kernel::*;
pub(crate) use kernel::*;
pub(crate) use row_arrangement::{LineBytes, RowArrangement};
pub(crate) use swizzle::ChunkSwizzle;
pub(crate) use swizzle::swizzled_line;

pub use build::DimsBuilder;
pub use build::split;
pub use compaction::Compaction;
pub use dim::{Divisor, Offset, PhysicalAxisMap, Scale};
pub use geometry::LineMisfit;
pub use geometry::RuntimeGeometry;
pub use storage_partitioning::base::TileMisfit;
pub use storage_partitioning::base::{StorageLevels, StorageMisfit, StoragePartitioning};
pub use vector_tile::VectorTile;
