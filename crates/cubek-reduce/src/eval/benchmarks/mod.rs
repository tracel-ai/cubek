//! Benchmark catalogue for `cubek-reduce`.

mod benchmark;
mod category;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use category::ReduceCategory;
pub use correctness::ReduceCorrectness;
pub use problem::{ReduceBenchKind, ReduceBenchPrecision, ReduceProblem, precisions, problems};
pub use strategy::strategies;
