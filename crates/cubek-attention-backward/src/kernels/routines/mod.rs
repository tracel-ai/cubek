//! The routines of attention's backward pass, one kernel each, one folder per scope: walks owned
//! by the queries and walks owned by the keys, the delta routine, and the casts out of the sums
//! some walks add into. The strategies in [`strategies`](super::strategies) launch them in order.

mod cast;
mod cube;
mod hypercube;
pub(crate) mod key_heads;
mod plane;
mod unit;

pub use cast::*;
pub use cube::*;
pub use hypercube::*;
pub use key_heads::KeyHeads;
pub use plane::*;
pub use unit::*;
