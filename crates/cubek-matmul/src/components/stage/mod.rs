// ! Performs tiled matrix multiplication using shared memory.
// ! Manages unit/plane coordination

mod matmul;

mod base;
mod event_listener;
mod memory;
// `stage_matmul` carries the new instance enum (`StageMatmul`,
// `StageMatmulKind`, `PartitionedStageMatmul`). Exposed as a sub-module
// during PR 5 to avoid a glob clash with the still-alive `StageMatmul`
// trait re-exported from `base`. PR 6 deletes the trait and glob-re-exports
// from here.
pub mod stage_matmul;

pub use base::*;
pub use event_listener::*;
pub use matmul::*;
pub use memory::*;
