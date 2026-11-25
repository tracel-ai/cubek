//! The benchmark catalogue of attention's backward pass: training problems, timed for each
//! strategy's lead. Gated behind the `benchmarks` feature; the top-level
//! `benchmarks` crate re-exports [`benchmarks::Category`].

pub mod benchmarks;
