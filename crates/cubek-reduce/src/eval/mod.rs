#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;

#[cfg(feature = "benchmarks")]
pub mod benchmarks;

mod workspace;

pub use workspace::EvalWorkspace;
