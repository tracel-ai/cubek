use std::path::PathBuf;
use std::time::Duration;

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

/// What `correctness_against_reference` should do with the reference file.
///
/// `WriteReference` is used by the trusted-commit pass: run the kernel once
/// and dump its output to `path`. `ValidateReference` is the current-commit
/// pass: run the kernel, compare its output against the file at `path`.
#[derive(Debug, Clone)]
pub enum ReferenceMode {
    WriteReference { path: PathBuf },
    ValidateReference { path: PathBuf },
}

/// Outcome of a correctness check. Mirrors `cubek_test_utils::ValidationResult`
/// at the registry boundary so the trait doesn't leak the test-utils dep
/// signature to consumers (the runner, tuner) — the registry crate itself can
/// still depend on test-utils.
#[derive(Debug, Clone)]
pub enum CorrectnessOutcome {
    /// Output matches the reference (or the CPU reference, depending on mode).
    Pass,
    /// Output does not match. The string describes how (max |Δ|, worst index,
    /// first mismatches, …) for the UI to render verbatim.
    Fail(String),
    /// Validation could not be completed (e.g. dtype not supported by the
    /// reference path, missing reference file). Treat as actionable: the user
    /// might need to regenerate the reference or pick a different problem.
    Error(String),
    /// `WriteReference` mode: the reference file was produced. No comparison
    /// happened (by design).
    WroteReference,
}

/// One benchmark category exposed to the tuner. Each category lives in its own
/// module (`attention`, `gemm`, ...) and exposes a unit struct that implements
/// this trait. `all()` returns every category the crate ships.
pub trait BenchmarkCategory: Sync {
    /// Stable identifier — persisted in tuner-results history. Don't rename.
    fn id(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn strategies(&self) -> Vec<ItemDescriptor>;
    fn problems(&self) -> Vec<ItemDescriptor>;
    fn run(
        &self,
        strategy_id: &str,
        problem_id: &str,
        num_samples: usize,
    ) -> Result<RunSamples, String>;

    /// Run the kernel once with seeded inputs and compare against an in-process
    /// CPU reference. Returns `None` when the category has no CPU reference
    /// implementation (e.g. memcpy_async, contiguous, unary) — the tuner UI
    /// hides those from the correctness page.
    ///
    /// `epsilon` is the tolerance to forward to `assert_equals_approx`.
    /// Default impl returns `None` so categories opt in as their references
    /// land.
    fn correctness_cpu(
        &self,
        _strategy_id: &str,
        _problem_id: &str,
        _epsilon: f32,
    ) -> Option<Result<CorrectnessOutcome, String>> {
        None
    }

    /// Run the kernel once with seeded inputs and either write its output to
    /// `mode.path` (WriteReference) or compare its output against the file at
    /// `mode.path` (ValidateReference). Inputs must be deterministic so two
    /// commits produce the same input bits.
    ///
    /// Returns `None` when the category doesn't expose a CPU-equivalent
    /// reference. We use the same predicate as `correctness_cpu` because the
    /// trusted-commit comparison is only meaningful when the kernel has a
    /// "ground truth" pinned somewhere — otherwise both runs could be wrong
    /// the same way and we'd silently agree.
    fn correctness_against_reference(
        &self,
        _strategy_id: &str,
        _problem_id: &str,
        _mode: ReferenceMode,
        _epsilon: f32,
    ) -> Option<Result<CorrectnessOutcome, String>> {
        None
    }
}

/// Every benchmark category compiled into this build of the registry. The
/// tuner-runner enumerates this slice — adding a new category is a single
/// `&Category` entry here, no changes needed in the tuner.
pub fn all() -> &'static [&'static dyn BenchmarkCategory] {
    &[
        &crate::attention::Category,
        &crate::contiguous::Category,
        &crate::conv2d::Category,
        &crate::fft::Category,
        &crate::gemm::Category,
        &crate::gemv::Category,
        &crate::memcpy_async::Category,
        &crate::quantized_matmul::Category,
        &crate::reduce::Category,
        &crate::unary::Category,
    ]
}
