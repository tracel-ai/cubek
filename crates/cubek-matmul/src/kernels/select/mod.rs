//! Which kernel runs a product: what the selector is asked about ([`problem`]), the rules
//! ([`selector`]) and the strategy that pins or measures. The tuning key the tables measure by is
//! [`tune`](super::tune)'s.

pub mod packed;
pub mod problem;
pub mod selector;
mod strategy;

pub use strategy::*;

#[cfg(test)]
mod tests;
