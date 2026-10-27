//! The attention microkernels: the queries over the keys and values,
//! `softmax(query · keyᵀ · scale) · value`, a folder per scope ([`unit`](mod@unit), [`plane`],
//! [`cube`], [`hypercube`]). Beside them, the trait a routine reads them through, the traits a cube
//! holds its planes through, and what every attention microkernel shares under them: the axes, the
//! operands, the work item, the partitioning, the staging count, how a cube lays its stages out,
//! and which keys a query sees. Where workers split a row's keys, their partials are merged by
//! [`split_keys_merge`](crate::split_keys_merge).

mod attention_microkernel;
mod axes;
pub mod cube;
pub mod hypercube;
mod operand;
mod partitioning;
pub mod plane;
mod stage_layout;
mod staging_count;
pub mod unit;
mod visibility;
mod visible_keys;
mod work_item;

pub use attention_microkernel::*;
pub use axes::*;
pub use operand::*;
pub use partitioning::*;
pub use stage_layout::*;
pub use staging_count::*;
pub use visibility::*;
pub use visible_keys::*;
pub use work_item::*;
