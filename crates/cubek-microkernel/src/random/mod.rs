//! The random microkernels, one folder per scope ([`unit`](mod@unit), [`plane`]), and what every
//! one of them shares: the generator a unit advances, the distributions a draw turns its values
//! into, and the transcendentals those draws evaluate.
//!
//! A unit owns one generator, seeded for its position in the launch, and draws a fixed number of
//! lines from it. What tells the scopes apart is only which lines of the output a unit's draws
//! land on: a run of its own, or lines interleaved with the other lanes of its plane.

mod bernoulli;
mod distribution;
mod interval;
mod normal;
pub mod plane;
pub mod polynomial;
mod state;
mod uniform;
pub mod unit;

pub use bernoulli::*;
pub use distribution::*;
pub use interval::*;
pub use normal::*;
pub use state::*;
pub use uniform::*;
