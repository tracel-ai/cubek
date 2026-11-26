//! Benchmark catalogue for `cubek-reduce`.

mod benchmark;
mod category;
mod correctness;
mod problem;
mod results;
mod strategy;

pub use category::ReduceCategory;
/// [`ReduceCategory`] by the name the benchmark registry still reaches it by.
pub use category::ReduceCategory as Category;
pub use problem::{ReduceBenchKind, ReduceBenchPrecision, ReduceProblem};
pub use results::BROKEN_LAUNCH;
