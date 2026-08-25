//! Benchmark catalogue for the depthwise routine.
//!
//! Separate from the `conv2d` category next door because it measures a different question. That
//! one sweeps convolution *strategies* over generic shapes; this one holds the strategy fixed at
//! "the depthwise routine" and sweeps its [`DepthwiseTiling`](crate::DepthwiseTiling) over the
//! shapes one real encoder runs — which is where a depthwise kernel's cost actually lives.

mod benchmark;
mod problem;

pub use benchmark::bench;
pub use problem::{DepthwiseProblem, blocks_running, problems, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::DepthwiseTiling;

/// What to run a problem under: the tiling the routine picks for itself, or a stated one.
///
/// The routine's own choice is a catalogue entry rather than a number computed on the side
/// because it is the one that ships. A sweep that only timed stated tilings would leave the
/// question the sweep exists to answer — is the rule picking the right one — to arithmetic done
/// somewhere else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthwiseStrategy {
    /// [`DepthwiseTiling::for_problem`], which is what [`launch_depthwise`](crate::launch_depthwise)
    /// runs.
    Routine,
    Fixed(DepthwiseTiling),
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = DepthwiseProblem;
    type Strategy = DepthwiseStrategy;

    fn id(&self) -> &'static str {
        "depthwise"
    }

    fn label(&self) -> &'static str {
        "Depthwise conv2d"
    }

    fn problems(&self) -> Vec<CatalogEntry<DepthwiseProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<DepthwiseStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &DepthwiseStrategy,
        problem: &DepthwiseProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
