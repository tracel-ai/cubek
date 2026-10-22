//! The cube scope: a cube owns a box of queries of some heads of the group, each of its planes
//! owning its queries of one head. A cube stages its keys and values for its planes, or its planes
//! read them where they lie.

mod base;
mod shuffle;
mod staged;

pub use base::*;
pub use shuffle::*;
pub use staged::*;
