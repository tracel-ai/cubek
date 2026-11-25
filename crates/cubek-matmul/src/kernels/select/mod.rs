//! Which kernel runs a product: what the selector is asked about ([`problem`]), the rules
//! ([`selector`]), the race a measured product goes to ([`MatmulAutotune`]) and the strategy that
//! pins or measures. What tells two products apart for a caller that measures them is
//! [`MatmulTuneKey`](super::MatmulTuneKey).

pub mod packed;
pub mod problem;
pub mod selector;

mod autotune;
mod strategy;

pub use autotune::*;
pub use strategy::*;

#[cfg(test)]
mod plans;
#[cfg(test)]
mod stub;
#[cfg(test)]
mod tests;

#[cfg(test)]
use stub::stub_measured;
