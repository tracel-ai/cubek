//! The `recurrence_walk` category: [`problems`] by [`strategies`], each row one walk in `f32`.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{RecurrenceWalkSpec, problems};
pub use strategy::{RecurrenceWalkStrategy, strategies};

use cubecl::benchmark::TimingMethod;
use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = RecurrenceWalkSpec;
    type Strategy = RecurrenceWalkStrategy;

    fn id(&self) -> &'static str {
        "recurrence_walk"
    }

    fn label(&self) -> &'static str {
        "Recurrence walk"
    }

    fn problems(&self) -> Vec<CatalogEntry<RecurrenceWalkSpec>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<RecurrenceWalkStrategy>> {
        strategies()
    }

    fn timing_method(&self) -> TimingMethod {
        cubek_test_utils::timing_method(TimingMethod::Device)
    }

    fn bench(
        &self,
        strategy: &RecurrenceWalkStrategy,
        spec: &RecurrenceWalkSpec,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, spec, num_samples)
    }
}
