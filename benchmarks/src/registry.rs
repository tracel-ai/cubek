use std::time::Duration;

use cubek_test_utils::HostData;

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

/// Each category lives in its own module (`attention`, `gemm`, ...) and
/// exposes a unit struct that implements this trait. `all()`
/// returns every category the crate ships.
///
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

    /// Run `strategy_id` on `problem_id` with the given seeded inputs and
    /// return its output as a [`HostData`]. `None` means the category doesn't
    /// expose a kernel result (e.g. memcpy_async — no semantic-level output).
    ///
    /// Both inputs and the resulting output must be deterministic under
    /// `(strategy_id, problem_id, seed_lhs, seed_rhs)` so the same call on two
    /// commits produces the same input bits and a directly-comparable output.
    fn kernel_result(
        &self,
        _strategy_id: &str,
        _problem_id: &str,
        _seed_lhs: u64,
        _seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        None
    }

    /// CPU-side ground-truth counterpart of [`Self::kernel_result`] for the
    /// same `(problem_id, seeds)`. `None` when the category has no
    /// CPU-equivalent reference (e.g. unary, contiguous).
    ///
    /// Same input bits as `kernel_result`, so the returned `HostData` is
    /// directly comparable elementwise.
    fn reference_result(
        &self,
        _problem_id: &str,
        _seed_lhs: u64,
        _seed_rhs: u64,
    ) -> Option<Result<HostData, String>> {
        None
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
        &crate::memcpy_async::Category,
        &crate::quantized_matmul::Category,
        &crate::reduce::Category,
        &crate::unary::Category,
    ]
}
