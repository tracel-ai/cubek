//! Mosaic: a CPU-first matmul built directly on the [`cubek_tile`] DSL, and a
//! laboratory for trying different operand
//! [`InnerLayout`](crate::definition::InnerLayout)s under one kernel.

mod kernel;

pub use kernel::*;
