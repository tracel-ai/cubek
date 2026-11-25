//! Which kernel an attention runs: what a caller can ask for ([`AttentionOpStrategy`]) and the
//! pure rule that resolves it ([`selector`]). What tells two attentions apart for a caller that
//! measures them is [`AttentionKey`](super::AttentionKey).

#[cfg(test)]
mod plans;
pub mod problem;
pub mod selector;
mod strategy;
#[cfg(test)]
mod stub;

pub use strategy::*;
#[cfg(test)]
use stub::stub_measured;
