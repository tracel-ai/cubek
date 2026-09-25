//! Where an operand's axes live: [`Projection`] maps logical axes onto buffer dims, extent-free;
//! [`Geometry`] is the buffer's extents and strides; the in-kernel halves evaluate both.

mod buffer;
mod build;
mod compaction;
mod dim;
mod geometry;
mod in_kernel;
mod kernel;
mod projection;
mod storage_tiling;

pub(crate) use buffer::*;
pub use build::*;
pub use compaction::*;
pub use dim::*;
pub use geometry::*;
pub use in_kernel::*;
pub(crate) use kernel::*;
pub use projection::*;
pub use storage_tiling::*;
