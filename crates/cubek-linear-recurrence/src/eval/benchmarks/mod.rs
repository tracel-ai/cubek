//! The `gated_delta_walk` category: [`problems`] by [`strategies`], each row one walk in `f32`.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{GatedDeltaWalkSpec, problems};
pub use strategy::{GatedDeltaWalkStrategy, strategies};

use cubecl::benchmark::TimingMethod;
use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = GatedDeltaWalkSpec;
    type Strategy = GatedDeltaWalkStrategy;

    fn id(&self) -> &'static str {
        "gated_delta_walk"
    }

    fn label(&self) -> &'static str {
        "Gated delta walk"
    }

    fn problems(&self) -> Vec<CatalogEntry<GatedDeltaWalkSpec>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<GatedDeltaWalkStrategy>> {
        strategies()
    }

    fn timing_method(&self) -> TimingMethod {
        cubek_test_utils::timing_method(TimingMethod::Device)
    }

    fn bench(
        &self,
        strategy: &GatedDeltaWalkStrategy,
        spec: &GatedDeltaWalkSpec,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, spec, num_samples)
    }
}
