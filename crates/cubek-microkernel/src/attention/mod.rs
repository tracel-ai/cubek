//! The attention microkernels, a folder per problem: the [`output`] of the queries over the keys
//! and values, and the [`merge`] of partials of the same rows, each from a worker that walked its
//! own part of the keys. Beside them, the trait a routine reads them through, the traits a cube
//! holds its planes through, and what every attention microkernel shares under them: the axes,
//! the operands, where a plane shuffle's lines come from, the work item, the partitioning, the
//! staging count, how a cube lays its stages out, how partials of the same rows lie in slots, and
//! which keys a query sees.

mod attention_microkernel;
mod axes;
mod cube_plane_attention;
mod lines;
pub mod merge;
mod operand;
pub mod output;
mod partitioning;
mod slots;
mod stage_layout;
mod staging_count;
mod visibility;
mod visible_keys;
mod work_item;

pub use attention_microkernel::*;
pub use axes::*;
pub use cube_plane_attention::*;
pub use lines::*;
pub use operand::*;
pub use partitioning::*;
pub use slots::*;
pub use stage_layout::*;
pub use staging_count::*;
pub use visibility::*;
pub use visible_keys::*;
pub use work_item::*;
