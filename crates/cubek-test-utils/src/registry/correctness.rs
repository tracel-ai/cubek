use crate::{HostData, KernelResultError, Progress};

/// Optional correctness surface for a category. Both methods take a `seeds`
/// slice instead of fixed `seed_lhs`/`seed_rhs` so unary ops use just
/// `seeds[0]` and future ops with more inputs can take more seeds without
/// churning the trait.
///
/// Convention: `seeds[0]` is the lhs seed, `seeds[1]` (when present) the rhs
/// seed. The registry's `BenchmarkCategory` adapter always passes a 2-element
/// slice today.
pub trait Correctness: Sync {
    type Problem;
    type Strategy;

    /// Run `strategy` on `problem` with the given seeded inputs and return its
    /// output as [`HostData`]. Output must be deterministic under
    /// `(strategy, problem, seeds)` so the same call on two commits produces
    /// directly-comparable bits.
    ///
    /// # Errors
    ///
    /// [`KernelResultError::Refused`] where the strategy's routine refuses the problem and says
    /// why, [`KernelResultError::NotRun`] where the kernel did not run for another reason.
    fn kernel_result(
        &self,
        strategy: &Self::Strategy,
        problem: &Self::Problem,
        seeds: &[u64],
    ) -> Result<HostData, KernelResultError>;

    /// CPU-side ground-truth counterpart of [`Self::kernel_result`] for the
    /// same `(problem, seeds)`. `progress`, when provided, is `set_total`'d
    /// to the output-write count and bumped once per write so callers can
    /// stream a progression bar.
    fn reference_result(
        &self,
        problem: &Self::Problem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String>;
}
