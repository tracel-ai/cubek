//! The cast routines: a gradient summed in `f32` by cubes that each added their share, cast into
//! the gradient's own element once every cube is done.

mod kernel;
mod key_gradient;
mod query_gradient;

pub use key_gradient::*;
pub use query_gradient::*;
