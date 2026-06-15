pub mod batch;
pub mod global;
pub mod stage;
pub mod tile;

// Internal-only — external crates import this directly from cubek-std.
pub(crate) use cubek_std::CubeDimResource;
