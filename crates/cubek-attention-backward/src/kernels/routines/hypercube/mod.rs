//! The hypercube routines: walks owned by the keys that the hypercube scope extends to the
//! query's gradient. Every cube owns a box of keys of one key head as the cube staged routines'
//! walk owned by the keys does, and also adds its share of the gradient of each query it sees
//! into one `f32` sum every such cube adds into, atomically, once a stage of queries: its planes,
//! which split its keys, sum their shares in shared memory first. Five multiplications a tile
//! instead of the two walks' seven.
//!
//! The atomic adds land in no set order, so the query's gradient may differ from one run to the
//! next in its last bits.

mod base;
mod gradient_sum;
mod kernel;
mod key_gradient_query_sum;

pub use base::*;
pub use gradient_sum::*;
pub use key_gradient_query_sum::*;
