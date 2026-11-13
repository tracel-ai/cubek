//! The routines of attention's backward pass, one kernel each, one folder per scope: walks owned
//! by the queries and walks owned by the keys, the delta routine, and the casts out of the sums
//! some walks add into. The strategies in [`strategies`](super::strategies) launch them in order.

mod cast;
mod cube;
pub(crate) mod heads_apart;
mod hypercube;
mod plane;
mod unit;

pub use cast::*;
pub use cube::*;
pub use hypercube::*;
pub use plane::*;
pub use unit::*;
