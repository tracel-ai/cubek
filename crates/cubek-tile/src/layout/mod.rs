//! Where an operand's axes live: [`Projection`] maps logical axes onto buffer dims, extent-free;
//! [`Geometry`] is the buffer's extents and strides; the in-kernel halves evaluate both. A stage's
//! block rows are placed across the banks by [`RowPlacement`] and [`ChunkSwizzle`].

mod buffer;
mod build;
mod compaction;
mod dim;
mod geometry;
mod in_kernel;
mod kernel;
mod placement;
mod projection;
mod storage_tiling;
mod swizzle;

pub(crate) use buffer::*;
pub use build::*;
pub use compaction::*;
pub use dim::*;
pub use geometry::*;
pub use in_kernel::*;
pub(crate) use kernel::*;
pub(crate) use placement::{LineBytes, RowPlacement, placed_digits, stage_offset};
pub use projection::*;
pub use storage_tiling::*;
pub(crate) use swizzle::ChunkSwizzle;
pub(crate) use swizzle::swizzled_line;
