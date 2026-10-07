//! Where an operand's axes live: [`Projection`](crate::Projection) maps logical axes onto buffer dims,
//! [`Geometry`](crate::Geometry) holds extents and strides, [`StoragePartitioning`] how values are tiled.

pub(crate) mod buffer_layout;
pub(crate) mod compaction;
pub(crate) mod compaction_step;
pub(crate) mod dims_builder;
pub(crate) mod geometry;
pub(crate) mod physical_axis_map;
pub(crate) mod projection;
pub(crate) mod projection_in_kernel;
pub(crate) mod row_arrangement;
pub(crate) mod runtime_map;
pub(crate) mod storage_partitioning;
pub(crate) mod swizzle;
pub(crate) mod vector_tile;
pub(crate) mod witness;

pub(crate) use buffer_layout::*;
pub(crate) use compaction_step::*;
pub(crate) use physical_axis_map::*;
pub(crate) use projection_in_kernel::*;
pub(crate) use row_arrangement::{LineBytes, RowArrangement};
pub(crate) use runtime_map::*;
pub(crate) use swizzle::ChunkSwizzle;
pub(crate) use swizzle::swizzled_line;
pub(crate) use witness::Witness;

pub use compaction::Compaction;
pub use dims_builder::{DimsBuilder, split};
pub use geometry::LineMisfit;
pub use geometry::RuntimeGeometry;
pub use physical_axis_map::{Divisor, Offset, PhysicalAxisMap, Scale};
pub use storage_partitioning::{StorageLevels, StorageMisfit, StoragePartitioning, TileMisfit};
pub use vector_tile::VectorTile;
