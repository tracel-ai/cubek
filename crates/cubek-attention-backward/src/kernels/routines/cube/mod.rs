//! The cube staged routines: every cube owns a box of queries of some heads of a group, or of
//! keys of one key head, its planes each on a matrix instruction. A cube stages what its walk
//! holds still once, and each stage of what it walks once, for all its planes, met on
//! `sync_cube`; each plane holds its own windows.
//!
//! What the plane routines' planes each read again, the cube reads once: the keys and values every
//! head and every plane along the queries read in the walk owned by the queries, and the queries
//! every plane along the keys reads in the walk owned by the keys.

mod base;
mod kernel;
mod key_gradient;
mod key_gradient_sum;
mod query_gradient;

pub use base::*;
pub use key_gradient::*;
pub use key_gradient_sum::*;
pub use query_gradient::*;
