//! Benchmark / correctness registry traits shared across kernel crates.
//!
//! Each kernel crate exposes a `pub mod benchmarks` (gated behind its
//! `benchmarks` cargo feature) that defines a [`Category`] over the kernel's
//! own problem and strategy types. The top-level `benchmarks` crate then
//! collects those into a single `all()` slice for harnesses (Cargo benches)
//! and for the tuner-runner.

use std::time::Duration;

use cubecl::benchmark::TimingMethod;
use cubecl::prelude::*;
use cubecl::std::throughput::measure_peak_throughput;
use cubecl::throughput::{
    self, MemoryAccess, ResourceBound, ThroughputKey, ThroughputMode, score_resources,
};
use cubecl::{Runtime, TestRuntime, client::ComputeClient};

use crate::{HostData, Progress};

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

/// Which resource a bench row's binding measurement is judged against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    Compute,
    Read,
    Write,
}

/// The resource that bound a run's duration, the slowest of the declared
/// resources even running flat out at its own peak, alongside how fast the
/// run actually achieved it. Mirrors the selection rule in
/// [`throughput::binding_achieved`].
#[derive(Debug, Clone, Copy)]
pub struct Binding {
    pub resource: ResourceKind,
    pub achieved_per_s: f64,
    pub peak_per_s: f64,
    pub fraction_of_peak: f64,
}

#[derive(Debug, Clone)]
pub struct RunSamples {
    pub durations: Vec<Duration>,
    /// TFLOPS achieved against the measured compute peak. `None` when the
    /// category's [`CategoryWork`] declares no compute (memcpy, contiguous,
    /// random, ...).
    pub tflops: Option<f64>,
    /// The resource that bound this run's duration. `None` when the category
    /// declares no [`CategoryWork`] for this problem, or none of the declared
    /// resources had a usable peak measurement.
    pub binding: Option<Binding>,
}

impl RunSamples {
    pub fn new(durations: Vec<Duration>) -> Self {
        Self {
            durations,
            tflops: None,
            binding: None,
        }
    }

    /// Median sample duration in seconds, or `None` when there are no samples
    /// or the median is zero (which would divide a throughput into NaN/inf).
    fn median_secs(&self) -> Option<f64> {
        if self.durations.is_empty() {
            return None;
        }
        let mut ns: Vec<u128> = self.durations.iter().map(|d| d.as_nanos()).collect();
        ns.sort_unstable();
        let median_secs = ns[ns.len() / 2] as f64 / 1e9;
        (median_secs > 0.0).then_some(median_secs)
    }
}

/// What a category's `(strategy, problem)` run honestly moves and computes:
/// bytes per direction and, where meaningful, a compute op count. The blanket
/// [`BenchmarkCategory`] adapter turns this into [`ResourceBound`]s against
/// measured device peaks. A category builds this from `problem` alone
/// (shapes and dtypes), never from a measurement of its own run.
#[derive(Debug, Clone, Copy)]
pub struct CategoryWork {
    /// Compute operations the run performs, counted the way
    /// [`ThroughputMode::ComputeDirect`] counts them (a multiply-add is 2).
    /// `0` when the category has no honest count (an elementwise
    /// transcendental, an FFT). The adapter then skips the compute bound.
    pub compute_ops: usize,
    /// The dtype the compute runs in. Ignored when `compute_ops` is `0`.
    pub dtype: ElemType,
    /// Bytes read from global memory.
    pub bytes_read: usize,
    /// Bytes written to global memory.
    pub bytes_written: usize,
}

impl CategoryWork {
    /// The declared resources as `(kind, bound)` pairs, peaks measured fresh
    /// on `client`. `measure_peak_throughput` caches per `(device, key)`, so
    /// repeated calls for the same working-set size cost nothing after the
    /// first.
    fn resources(&self, client: &ComputeClient<TestRuntime>) -> Vec<(ResourceKind, ResourceBound)> {
        let mut out = Vec::with_capacity(3);

        if self.compute_ops > 0 {
            let key = ThroughputKey {
                mode: ThroughputMode::ComputeDirect { dtype: self.dtype },
            };
            let peak = measure_peak_throughput(client, key).ops_per_s();
            out.push((
                ResourceKind::Compute,
                ResourceBound {
                    amount: self.compute_ops,
                    peak_per_s: peak,
                },
            ));
        }
        if self.bytes_read > 0 {
            let key = ThroughputKey {
                mode: ThroughputMode::MemoryWorkingSet {
                    access: MemoryAccess::Read,
                    bytes: self.bytes_read as u64,
                },
            };
            let peak = measure_peak_throughput(client, key).bytes_per_s(&key);
            out.push((
                ResourceKind::Read,
                ResourceBound {
                    amount: self.bytes_read,
                    peak_per_s: peak,
                },
            ));
        }
        if self.bytes_written > 0 {
            let key = ThroughputKey {
                mode: ThroughputMode::MemoryWorkingSet {
                    access: MemoryAccess::Write,
                    bytes: self.bytes_written as u64,
                },
            };
            let peak = measure_peak_throughput(client, key).bytes_per_s(&key);
            out.push((
                ResourceKind::Write,
                ResourceBound {
                    amount: self.bytes_written,
                    peak_per_s: peak,
                },
            ));
        }

        out
    }
}

/// Scores `work` against measured device peaks and fills `samples`'s
/// `tflops`/`binding` from the median sample duration. Left unfilled (and
/// `samples` returned as is) when there are no samples or the median duration
/// is zero.
fn score(mut samples: RunSamples, work: &CategoryWork) -> RunSamples {
    let Some(median_secs) = samples.median_secs() else {
        return samples;
    };

    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let resources = work.resources(&client);
    let bounds: Vec<ResourceBound> = resources.iter().map(|(_, bound)| *bound).collect();
    let scores = score_resources(Duration::from_secs_f64(median_secs), &bounds);

    if let Some(idx) = resources
        .iter()
        .position(|(kind, _)| *kind == ResourceKind::Compute)
    {
        samples.tflops = Some(scores[idx].achieved_per_s / 1e12);
    }

    if let Some(best) = throughput::binding_achieved(&scores) {
        let idx = scores
            .iter()
            .position(|candidate| candidate == best)
            .expect("binding_achieved returns an element of scores");
        let (resource, bound) = resources[idx];
        samples.binding = Some(Binding {
            resource,
            achieved_per_s: best.achieved_per_s,
            peak_per_s: bound.peak_per_s,
            fraction_of_peak: best.fraction_of_peak,
        });
    }

    samples
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

    /// The work `problem` honestly represents, for the blanket adapter to score
    /// against measured device peaks. Built from the problem's shapes and
    /// dtypes alone, the same for every strategy that runs it. `None` (the
    /// default) when no honest count exists for this problem; its run then
    /// reports plain durations, unscored.
    fn work(&self, _problem: &Self::Problem) -> Option<CategoryWork> {
        None
    }

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
        let samples = Category::bench(self, &strategy.value, &problem.value, num_samples)?;
        Ok(match Category::work(self, &problem.value) {
            Some(work) => score(samples, &work),
            None => samples,
        })
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
