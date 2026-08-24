//! Benchmark / correctness registry traits shared across kernel crates.
//!
//! Each kernel crate exposes a `pub mod benchmarks` (gated behind its
//! `benchmarks` cargo feature) that defines a [`Category`] over the kernel's
//! own problem and strategy types. The top-level `benchmarks` crate then
//! collects those into a single `all()` slice for harnesses (Cargo benches)
//! and for the tuner-runner.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use cubecl::benchmark::TimingMethod;
use cubecl::prelude::*;
use cubecl::std::throughput::measure_peak_throughput;
use cubecl::throughput::{
    self, MemoryAccess, ResourceBound, ThroughputKey, ThroughputMode, score_resources,
};
use cubecl::{Runtime, TestRuntime, client::ComputeClient};

use crate::{HostData, Progress};

/// The client every category scores against: `measure_peak_throughput` is
/// always run on `<TestRuntime as Runtime>::Device::default()`, so the
/// process-wide peak memo below can key on [`ThroughputKey`] alone.
fn client() -> ComputeClient<TestRuntime> {
    let device = <TestRuntime as Runtime>::Device::default();
    <TestRuntime as Runtime>::client(&device)
}

/// Process-wide memo of measured peaks, keyed by [`ThroughputKey`].
///
/// `measure_peak_throughput` allocates and primes a probe buffer before it
/// consults its own per-device cache, so calling it between two timed rows
/// moves real memory even when the peak is already known.
static PEAKS: LazyLock<Mutex<HashMap<ThroughputKey, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Looks up `key`'s peak rate for `kind` in [`PEAKS`], measuring and caching
/// it on a miss. Never re-enters itself, so holding the lock across the
/// measurement cannot deadlock.
fn peak_per_s(kind: ResourceKind, key: ThroughputKey) -> f64 {
    let mut peaks = PEAKS.lock().expect("peaks mutex is not poisoned");
    *peaks.entry(key).or_insert_with(|| {
        let value = measure_peak_throughput(&client(), key);
        match kind {
            ResourceKind::Compute => value.ops_per_s(),
            ResourceKind::Read | ResourceKind::Write => value.bytes_per_s(&key),
        }
    })
}

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
    /// Every resource this work declares, one entry per non-zero field, as
    /// `(kind, amount, key)` in the order the adapter scores them: Compute,
    /// Read, Write.
    fn declared_resources(&self) -> Vec<(ResourceKind, usize, ThroughputKey)> {
        let mut out = Vec::with_capacity(3);

        if self.compute_ops > 0 {
            let key = ThroughputKey {
                mode: ThroughputMode::ComputeDirect { dtype: self.dtype },
            };
            out.push((ResourceKind::Compute, self.compute_ops, key));
        }
        if self.bytes_read > 0 {
            let key = ThroughputKey {
                mode: ThroughputMode::MemoryWorkingSet {
                    access: MemoryAccess::Read,
                    bytes: self.bytes_read as u64,
                },
            };
            out.push((ResourceKind::Read, self.bytes_read, key));
        }
        if self.bytes_written > 0 {
            let key = ThroughputKey {
                mode: ThroughputMode::MemoryWorkingSet {
                    access: MemoryAccess::Write,
                    bytes: self.bytes_written as u64,
                },
            };
            out.push((ResourceKind::Write, self.bytes_written, key));
        }

        out
    }

    /// The declared resources as `(kind, bound)` pairs, peaks pulled from the
    /// process-wide memo (see [`peak_per_s`]).
    fn resources(&self) -> Vec<(ResourceKind, ResourceBound)> {
        self.declared_resources()
            .into_iter()
            .map(|(kind, amount, key)| {
                (
                    kind,
                    ResourceBound {
                        amount,
                        peak_per_s: peak_per_s(kind, key),
                    },
                )
            })
            .collect()
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

    let resources = work.resources();
    let (tflops, binding) = score_bounds(median_secs, &resources);
    samples.tflops = tflops;
    samples.binding = binding;

    samples
}

/// The pure half of [`score`]: given already-measured resource bounds, scores
/// them at `median_secs` and picks the binding one. Split out so the selection
/// logic (which resource binds, what `tflops` reads) is testable against
/// fabricated bounds instead of a device's actual measured peaks.
fn score_bounds(
    median_secs: f64,
    resources: &[(ResourceKind, ResourceBound)],
) -> (Option<f64>, Option<Binding>) {
    let bounds: Vec<ResourceBound> = resources.iter().map(|(_, bound)| *bound).collect();
    let scores = score_resources(Duration::from_secs_f64(median_secs), &bounds);

    let tflops = resources
        .iter()
        .position(|(kind, _)| *kind == ResourceKind::Compute)
        .map(|idx| scores[idx].achieved_per_s / 1e12);

    let binding = throughput::binding_achieved(&scores).map(|best| {
        let idx = scores
            .iter()
            .position(|candidate| candidate == best)
            .expect("binding_achieved returns an element of scores");
        let (resource, bound) = resources[idx];
        Binding {
            resource,
            achieved_per_s: best.achieved_per_s,
            peak_per_s: bound.peak_per_s,
            fraction_of_peak: best.fraction_of_peak,
        }
    });

    (tflops, binding)
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

    /// Measures and memoizes every distinct peak this category declares, so
    /// the first timed row never pays for a probe. Returns how many
    /// distinct keys it warmed.
    fn warm_peaks(&self) -> usize;

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

    fn warm_peaks(&self) -> usize {
        let mut seen = HashSet::new();
        for problem in Category::problems(self) {
            let Some(work) = Category::work(self, &problem.value) else {
                continue;
            };
            for (kind, _, key) in work.declared_resources() {
                if seen.insert(key) {
                    peak_per_s(kind, key);
                }
            }
        }
        seen.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn work(compute_ops: usize, bytes_read: usize, bytes_written: usize) -> CategoryWork {
        CategoryWork {
            compute_ops,
            dtype: f32::elem_type_native(),
            bytes_read,
            bytes_written,
        }
    }

    /// Zero-ops, zero-bytes work declares no resource: nothing to score, so the
    /// adapter's row falls back to plain durations.
    #[test]
    fn zero_work_declares_no_resources() {
        let resources = work(0, 0, 0).resources();
        assert!(resources.is_empty());

        let (tflops, binding) = score_bounds(1.0, &resources);
        assert_eq!(tflops, None);
        assert!(binding.is_none());
    }

    /// A read-only declaration produces exactly one `Read` bound, carrying the
    /// declared byte count as its `amount`; no `Write` or `Compute` entry.
    #[test]
    fn read_only_work_declares_a_single_read_resource() {
        let resources = work(0, 4096, 0).resources();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].0, ResourceKind::Read);
        assert_eq!(resources[0].1.amount, 4096);
    }

    /// A write-only declaration produces exactly one `Write` bound, mirroring
    /// the read-only case.
    #[test]
    fn write_only_work_declares_a_single_write_resource() {
        let resources = work(0, 0, 2048).resources();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].0, ResourceKind::Write);
        assert_eq!(resources[0].1.amount, 2048);
    }

    /// A mixed declaration produces all three bounds, each carrying its own
    /// field's amount, compute first.
    #[test]
    fn mixed_work_declares_compute_read_and_write_resources() {
        let resources = work(1_000_000, 4096, 1024).resources();
        assert_eq!(resources.len(), 3);
        assert_eq!(
            resources.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(),
            vec![
                ResourceKind::Compute,
                ResourceKind::Read,
                ResourceKind::Write
            ]
        );
        assert_eq!(resources[0].1.amount, 1_000_000);
        assert_eq!(resources[1].1.amount, 4096);
        assert_eq!(resources[2].1.amount, 1024);
    }

    /// Same-shaped work keys to the same `ThroughputKey`; a different shape does not.
    #[test]
    fn declared_resources_key_on_shape_not_identity() {
        let a = work(0, 4096, 0).declared_resources();
        let b = work(0, 4096, 0).declared_resources();
        let c = work(0, 8192, 0).declared_resources();

        assert_eq!(a[0].2, b[0].2);
        assert_ne!(a[0].2, c[0].2);

        let mut seen = HashSet::new();
        seen.insert(a[0].2);
        seen.insert(b[0].2);
        seen.insert(c[0].2);
        assert_eq!(seen.len(), 2);
    }

    /// No resources to score: `score_bounds` leaves both `tflops` and `binding`
    /// unset rather than dividing by an empty bound set.
    #[test]
    fn score_bounds_with_no_resources_is_unscored() {
        let (tflops, binding) = score_bounds(1.0, &[]);
        assert_eq!(tflops, None);
        assert!(binding.is_none());
    }

    /// A single `Read` resource at a known peak scores its own achieved rate
    /// and becomes the binding one, with no `tflops` (no `Compute` entry).
    #[test]
    fn score_bounds_read_only_binds_on_read() {
        let bound = ResourceBound {
            amount: 1000,
            peak_per_s: 500.0,
        };
        let (tflops, binding) = score_bounds(2.0, &[(ResourceKind::Read, bound)]);

        assert_eq!(tflops, None);
        let binding = binding.expect("a usable peak binds");
        assert_eq!(binding.resource, ResourceKind::Read);
        assert_eq!(binding.achieved_per_s, 500.0);
        assert_eq!(binding.peak_per_s, 500.0);
        assert_eq!(binding.fraction_of_peak, 1.0);
    }

    /// Same shape as the read-only case, on `Write`.
    #[test]
    fn score_bounds_write_only_binds_on_write() {
        let bound = ResourceBound {
            amount: 300,
            peak_per_s: 100.0,
        };
        let (_, binding) = score_bounds(1.0, &[(ResourceKind::Write, bound)]);

        let binding = binding.expect("a usable peak binds");
        assert_eq!(binding.resource, ResourceKind::Write);
        assert_eq!(binding.fraction_of_peak, 3.0);
    }

    /// Mirrors the upstream roofline test this adapter builds on: a compute
    /// bound that would finish quickly at its own peak, alongside a read bound
    /// that would still take longer even running flat out. The slower-at-peak
    /// resource binds even though it moves far more amount, and `tflops` still
    /// reads the compute entry's own achieved rate regardless of which one
    /// bound the run.
    #[test]
    fn score_bounds_mixed_picks_the_slower_at_peak_resource() {
        let duration = 1.0;
        let compute = ResourceBound {
            amount: 200,
            peak_per_s: 1000.0,
        }; // 0.2s at peak
        let read = ResourceBound {
            amount: 900_000,
            peak_per_s: 1_000_000.0,
        }; // 0.9s at peak, the binding one
        let write = ResourceBound {
            amount: 100_000,
            peak_per_s: 200_000.0,
        }; // 0.5s at peak

        let resources = vec![
            (ResourceKind::Compute, compute),
            (ResourceKind::Read, read),
            (ResourceKind::Write, write),
        ];
        let (tflops, binding) = score_bounds(duration, &resources);

        assert_eq!(tflops, Some(200.0 / 1e12));
        let binding = binding.expect("a usable peak binds");
        assert_eq!(binding.resource, ResourceKind::Read);
        assert_eq!(binding.fraction_of_peak, 0.9);
    }
}
