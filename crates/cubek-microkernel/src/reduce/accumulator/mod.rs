//! What a reduction accumulates its lines into ([`ReduceAccumulator`]), one module a carrier.

mod base;
mod family;
mod indexed;
mod line;
mod top_k;
mod top_values;
mod values;

pub use base::*;
pub use family::*;
pub use indexed::*;
pub(crate) use line::*;
pub use top_k::*;
pub use top_values::*;
pub use values::*;
