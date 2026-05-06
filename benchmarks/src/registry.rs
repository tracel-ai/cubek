use std::time::Duration;

use cubecl::benchmark::TimingMethod;
use cubek_test_utils::{HostData, Progress};

#[derive(Debug, Clone)]
pub struct ItemDescriptor {
    pub id: String,
    pub label: String,
}

impl ItemDescriptor {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
        }
    }
}

/// A catalogued problem or strategy: stable id, human label, and the typed
/// payload the category needs to actually run with. Categories build vectors
/// of these and the registry erases them to [`ItemDescriptor`] for callers
/// that only care about the id/label pair.
pub struct CatalogEntry<T> {
    pub id: String,
    pub label: String,
    pub value: T,
}

impl<T> CatalogEntry<T> {
    pub fn new(id: impl Into<String>, label: impl Into<String>, value: T) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            value,
        }
    }

    pub fn descriptor(&self) -> ItemDescriptor {
        ItemDescriptor {
            id: self.id.clone(),
            label: self.label.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunSamples {
    pub durations: Vec<Duration>,
    /// Optional throughput, e.g. TFLOPS for matmul/attention. `None` when the
    /// category doesn't have a meaningful FLOP count (memcpy, contiguous, ...).
    pub tflops: Option<f64>,
}

impl RunSamples {
    pub fn new(durations: Vec<Duration>) -> Self {
        Self {
            durations,
            tflops: None,
        }
    }

    pub fn with_tflops(mut self, tflops: f64) -> Self {
        self.tflops = Some(tflops);
        self
    }

    /// Convenience for matmul-style benches: turn a flop count into TFLOPS using
    /// the median sample duration. Returns `self` unchanged if there are no
    /// samples or the median is zero (avoiding NaN/inf in the dashboard).
    pub fn with_flops(self, flops: f64) -> Self {
        if self.durations.is_empty() {
            return self;
        }
        let mut ns: Vec<u128> = self.durations.iter().map(|d| d.as_nanos()).collect();
        ns.sort_unstable();
        let median_secs = ns[ns.len() / 2] as f64 / 1e9;
        if median_secs <= 0.0 {
            return self;
        }
        self.with_tflops(flops / median_secs / 1e12)
    }
}

/// Typed per-category definition. Implementors expose their problem and
/// strategy catalogues with the actual payloads attached, plus a typed
/// `bench` closure. The blanket impl below adapts to the string-keyed
/// [`BenchmarkCategory`] consumed by the public registry, so categories no
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

    /// Which timing method [`Self::bench`] uses internally — used by the bench
    /// runner to label its printed stats. Defaults to `System`; categories
    /// running on the device timing method (unary/contiguous/memcpy_async)
    /// override this.
    fn timing_method(&self) -> TimingMethod {
        TimingMethod::System
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
    fn kernel_result(
        &self,
        strategy: &Self::Strategy,
        problem: &Self::Problem,
        seeds: &[u64],
    ) -> Result<HostData, String>;

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

/// Public, string-keyed registry surface. Implemented automatically for any
/// type that implements [`Category`]; categories should implement `Category`
/// rather than this trait directly.
pub trait BenchmarkCategory: Sync {
    /// Stable identifier — persisted in tuner-results history. Don't rename.
    fn id(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn strategies(&self) -> Vec<ItemDescriptor>;
    fn problems(&self) -> Vec<ItemDescriptor>;
    fn timing_method(&self) -> TimingMethod {
        TimingMethod::System
    }
    fn run(
        &self,
        strategy_id: &str,
        problem_id: &str,
        num_samples: usize,
    ) -> Result<RunSamples, String>;

    /// `None` means the category doesn't expose a kernel result (e.g.
    /// memcpy_async — no semantic-level output).
    fn kernel_result(
        &self,
        _strategy_id: &str,
        _problem_id: &str,
        _seed_lhs: u64,
        _seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        None
    }

    /// `None` when the category has no CPU-equivalent reference (e.g. unary,
    /// contiguous).
    fn reference_result(
        &self,
        _problem_id: &str,
        _seed_lhs: u64,
        _seed_rhs: u64,
        _progress: Option<&Progress>,
    ) -> Option<Result<HostData, String>> {
        None
    }
}

impl<C: Category> BenchmarkCategory for C {
    fn id(&self) -> &'static str {
        Category::id(self)
    }

    fn label(&self) -> &'static str {
        Category::label(self)
    }

    fn strategies(&self) -> Vec<ItemDescriptor> {
        Category::strategies(self)
            .iter()
            .map(CatalogEntry::descriptor)
            .collect()
    }

    fn problems(&self) -> Vec<ItemDescriptor> {
        Category::problems(self)
            .iter()
            .map(CatalogEntry::descriptor)
            .collect()
    }

    fn timing_method(&self) -> TimingMethod {
        Category::timing_method(self)
    }

    fn run(
        &self,
        strategy_id: &str,
        problem_id: &str,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        let problems = Category::problems(self);
        let problem = problems
            .iter()
            .find(|e| e.id == problem_id)
            .ok_or_else(|| format!("unknown problem: {problem_id}"))?;
        let strategies = Category::strategies(self);
        let strategy = strategies
            .iter()
            .find(|e| e.id == strategy_id)
            .ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
        Category::bench(self, &strategy.value, &problem.value, num_samples)
    }

    fn kernel_result(
        &self,
        strategy_id: &str,
        problem_id: &str,
        seed_lhs: u64,
        seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        let correctness = Category::correctness(self)?;
        let problems = Category::problems(self);
        let problem = match problems.iter().find(|e| e.id == problem_id) {
            Some(p) => p,
            None => return Some(Err(format!("unknown problem: {problem_id}"))),
        };
        let strategies = Category::strategies(self);
        let strategy = match strategies.iter().find(|e| e.id == strategy_id) {
            Some(s) => s,
            None => return Some(Err(format!("unknown strategy: {strategy_id}"))),
        };
        Some(correctness.kernel_result(&strategy.value, &problem.value, &[seed_lhs, seed_rhs]))
    }

    fn reference_result(
        &self,
        problem_id: &str,
        seed_lhs: u64,
        seed_rhs: u64,
        progress: Option<&Progress>,
    ) -> Option<Result<HostData, String>> {
        let correctness = Category::correctness(self)?;
        let problems = Category::problems(self);
        let problem = match problems.iter().find(|e| e.id == problem_id) {
            Some(p) => p,
            None => return Some(Err(format!("unknown problem: {problem_id}"))),
        };
        Some(correctness.reference_result(&problem.value, &[seed_lhs, seed_rhs], progress))
    }
}

/// Every benchmark category compiled into this build of the registry.
pub fn all() -> &'static [&'static dyn BenchmarkCategory] {
    &[
        &crate::attention::Category,
        &crate::contiguous::Category,
        &crate::conv2d::Category,
        &crate::fft::Category,
        &crate::gemm::Category,
        &crate::gemv::Category,
        &crate::interpolate::Category,
        &crate::memcpy_async::Category,
        &crate::quantized_matmul::Category,
        &crate::reduce::Category,
        &crate::unary::Category,
    ]
}
