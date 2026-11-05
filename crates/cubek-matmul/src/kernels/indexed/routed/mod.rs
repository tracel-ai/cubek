//! The routed product: each lhs row against the slabs of a stacked rhs its routes name, launched as
//! the dense product of its batches of one row, the stack read through the routes. It is tuned by a
//! tuner of its own ([`launch_tuned`]), under a key of its own ([`RoutedMatmulKey`]).

mod settings;
mod tensors;
mod tune;

#[cfg(test)]
mod tests;

pub(crate) use settings::{RoutedCandidates, RoutedFloors};
pub use tensors::*;
pub use tune::*;
