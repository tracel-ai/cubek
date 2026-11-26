use cubecl::benchmark::TimingMethod;

use crate::{CatalogEntry, CategoryWork, Correctness, RunSamples};

/// Typed per-category definition. Implementors expose their problem and
/// strategy catalogues with the actual payloads attached, plus a typed
/// `bench` closure. The blanket impl below adapts to the string-keyed
/// [`BenchmarkCategory`](crate::BenchmarkCategory) consumed by the public registry, so categories no
/// longer have to write the lookup boilerplate.
pub trait Category: Sync {
    type Problem;
    type Strategy;

    /// Stable identifier — persisted in tuner-results history. Don't rename.
    fn id(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn problems(&self) -> Vec<CatalogEntry<Self::Problem>>;
    fn strategies(&self) -> Vec<CatalogEntry<Self::Strategy>>;
    fn bench(
        &self,
        strategy: &Self::Strategy,
        problem: &Self::Problem,
        num_samples: usize,
    ) -> Result<RunSamples, String>;

    /// The work `problem` represents, the same for every strategy that runs it.
    /// `None` leaves the run unscored, reporting plain durations.
    fn work(&self, _problem: &Self::Problem) -> Option<CategoryWork> {
        None
    }

    /// Which timing method [`Self::bench`] uses internally — used by the bench
    /// runner to label its printed stats. Defaults to `System`; categories
    /// running on the device timing method (unary/contiguous/memcpy_async)
    /// override this.
    fn timing_method(&self) -> TimingMethod {
        crate::timing_method(TimingMethod::System)
    }

    /// Override to expose seeded `kernel_result` / `reference_result`. Decoupled
    /// from `Category` itself so unary/contiguous/memcpy_async don't need
    /// `cfg`-gated stub methods.
    fn correctness(
        &self,
    ) -> Option<&dyn Correctness<Problem = Self::Problem, Strategy = Self::Strategy>> {
        None
    }
}
