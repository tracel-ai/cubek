//! Which kernel an attention runs: what a caller can ask for ([`AttentionOpStrategy`]) and the
//! pure rule that resolves it ([`selector`]). What tells two attentions apart for the tables is
//! [`tune`](super::tune)'s.

pub mod selector;
mod strategy;

pub use strategy::*;
