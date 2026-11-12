//! The strategies of attention's backward pass: routines launched in order, each one kernel,
//! which together write the three gradients. What autotune races.
//!
//! The unit, plane and cube staged strategies walk the tiles twice: once owned by the queries, for
//! their gradient and their dot products of the output with its gradient, then owned by the keys,
//! for theirs and their values'. The hypercube strategy walks them once, owned by the keys, after
//! the delta routine wrote the dot products, and casts the query's gradient out of the sum its
//! cubes added into. Where the key walk's heads are apart, a cast of the keys' sums follows it.

mod cube;
mod hypercube;
mod plane;
mod unit;

pub use cube::*;
pub use hypercube::*;
pub use plane::*;
pub use unit::*;
