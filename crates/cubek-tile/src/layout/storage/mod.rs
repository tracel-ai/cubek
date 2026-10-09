//! How an operand's values lie in memory: its extents and strides ([`Geometry`]), the nested tiles
//! they are laid down in ([`StoragePartitioning`]), and what one vector load brings
//! ([`VectorTile`]).

pub(crate) mod geometry;
pub(crate) mod partitioning;
pub(crate) mod vector_tile;

pub use geometry::{Geometry, LineMisfit, RuntimeGeometry};
pub use partitioning::{StorageLevels, StorageMisfit, StoragePartitioning, TileMisfit};
pub use vector_tile::VectorTile;
