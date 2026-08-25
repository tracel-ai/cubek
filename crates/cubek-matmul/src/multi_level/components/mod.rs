pub mod batch;
pub mod global;
pub mod stage;
pub mod tile;

// Internal-only: external crates name it `cubek_matmul::multi_level::CubeDimResource`.
pub(crate) use crate::multi_level::CubeDimResource;
