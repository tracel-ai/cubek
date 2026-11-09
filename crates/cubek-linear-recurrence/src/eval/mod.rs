//! The benchmark catalogue of the gated delta walk: a decode step and prompts of a many-headed
//! layer, timed for the cube routine's lead and for the tuned walk. Gated behind the `benchmarks`
//! feature; the top-level `benchmarks` crate re-exports [`benchmarks::Category`].

pub mod benchmarks;
