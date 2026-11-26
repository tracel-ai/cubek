//! Benchmark catalogue for `cubek-reduce`.

mod benchmark;
mod category;
mod correctness;
mod problem;
mod results;
mod strategy;

pub use category::ReduceCategory;
pub use problem::{ReduceBenchKind, ReduceBenchPrecision, ReduceProblem};
