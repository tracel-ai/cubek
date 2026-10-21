//! What a staged walk through `K` shares: the stage it fills, how its planes lie, what an estimate
//! counts of it, and how its cubes reach the output. The ring that holds its stages is
//! [`crate::microkernel::staging`]'s.

mod cube_write;
mod plane_layout;
mod stage;
mod staging_count;

pub use cube_write::*;
pub use plane_layout::*;
pub use stage::*;
pub use staging_count::*;
