//! What a reduction accumulates its lines into ([`ReduceAccumulator`]), one module a carrier.

mod base;
mod indexed;
mod line;
mod top_k;
mod top_values;
mod values;

pub use base::*;
pub(crate) use line::*;
