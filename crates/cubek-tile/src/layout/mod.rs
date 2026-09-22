//! Where an operand's axes live: [`Projection`] maps logical axes onto buffer dims, extent-free;
//! [`Geometry`] is the buffer's extents and strides; the in-kernel halves evaluate both.

mod base;
mod buffer;
mod build;
mod coefficients;
mod compaction;
mod dim;
mod geometry;
mod in_kernel;
mod kernel;
mod query;
mod storage_tiling;

pub use base::*;
pub(crate) use buffer::*;
pub use build::*;
pub use compaction::*;
pub use dim::*;
pub use geometry::*;
pub use in_kernel::*;
pub(crate) use kernel::*;
pub use storage_tiling::*;
