pub mod batch;
pub mod global;
pub mod stage;
pub mod tile_matmul;

mod resource;

// Internal-only — external crates import this directly from cubek-std.
pub(crate) use resource::*;
