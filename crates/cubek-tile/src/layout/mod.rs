//! Where an operand's axes live: [`Projection`] maps logical axes onto buffer dims (`projection`),
//! `storage` says how the values lie in memory, a `compaction` boxes a gathered sub-tile, and a
//! `row_arrangement` places a stage's lines.

pub(crate) mod compaction;
pub(crate) mod projection;
pub(crate) mod row_arrangement;
pub(crate) mod storage;

pub(crate) use compaction::*;
pub(crate) use projection::*;
pub(crate) use row_arrangement::*;

pub use compaction::Compaction;
pub use projection::{DimsBuilder, Divisor, Offset, PhysicalAxisMap, Scale, split};
pub use storage::{
    LineMisfit, RuntimeGeometry, StorageLevels, StorageMisfit, StoragePartitioning, TileMisfit,
    VectorTile,
};
