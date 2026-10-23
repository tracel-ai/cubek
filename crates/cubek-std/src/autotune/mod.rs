//! How a family races its candidates: the priority band that orders a round, and the autotune a
//! family builds its candidates into ([`Autotune`]), measured against the problem's roofline
//! ([`Roofline`]).

mod base;
mod group;
mod kernel;
mod priority;
mod roofline;

pub use base::*;
pub use group::*;
pub use priority::*;
pub use roofline::*;
