//! The benchmark catalogue of attention's backward pass: training problems, timed for each
//! routine's lead and for the tuned pass. Gated behind the `benchmarks` feature; the top-level
//! `benchmarks` crate re-exports [`benchmarks::Category`].

pub mod benchmarks;
