//! The interpolate routines: a cube routine reading the input where it lies, and one staging each
//! block's tap window in shared memory.

mod cube;
mod cube_staged;

pub use cube::*;
pub use cube_staged::*;
