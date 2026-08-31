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
use cubecl::std::throughput::{measure_memory_curve, measure_peak_throughput};
use cubecl::throughput::{
    self, MemoryAccess, MemoryCurve, ResourceBound, ThroughputKey, ThroughputMode, score_resources,
};
use cubecl::{Runtime, TestRuntime, client::ComputeClient};

use crate::{HostData, Progress};

/// The client every category scores against: `measure_peak_throughput` is
/// always run on `<TestRuntime as Runtime>::Device::default()`, so the
/// process-wide peak memo below can key on [`ThroughputKey`] alone.
pub fn client() -> ComputeClient<TestRuntime> {
    let device = <TestRuntime as Runtime>::Device::default();
    <TestRuntime as Runtime>::client(&device)
}

/// Process-wide memo of measured peaks.
///
/// `measure_peak_throughput` primes a probe buffer before consulting its own
/// cache, so calling it between two timed rows moves real memory regardless.
static PEAKS: LazyLock<Mutex<HashMap<ThroughputKey, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Process-wide memo of measured memory curves, one per access.
///
/// A curve answers every working set from one sweep, so a run asks the device
/// once per access rather than once per distinct size a problem declares.
static CURVES: LazyLock<Mutex<HashMap<MemoryAccess, MemoryCurve>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The rate a kernel moving `bytes` in the direction `access` names can reach,
/// read off [`CURVES`] and measuring the curve on a miss.
///
/// Zero when the sweep measured nothing, which leaves the resource unscored
/// rather than dividing by a ceiling that does not exist.
fn curve_ceiling(access: MemoryAccess, bytes: usize) -> f64 {
    let mut curves = CURVES.lock().expect("curves mutex is not poisoned");
    let curve = curves
        .entry(access)
        .or_insert_with(|| measure_memory_curve(&client(), access));

    curve.ceiling_at(bytes as u64).unwrap_or(0.0)
}

/// Looks up `key`'s peak rate in [`PEAKS`], measuring and caching it on a
/// miss. Never re-enters itself, so holding the lock across the measurement
/// cannot deadlock.
///
/// The unit comes from the key's own mode rather than from the caller, so a
/// memo keyed on the key alone cannot serve one resource's rate to another.
fn peak_per_s(key: ThroughputKey) -> f64 {
    let mut peaks = PEAKS.lock().expect("peaks mutex is not poisoned");
    *peaks.entry(key).or_insert_with(|| {
        let value = measure_peak_throughput(&client(), key);
        match key.mode {
            ThroughputMode::ComputeDirect { .. } | ThroughputMode::ComputeCmma { .. } => {
                value.ops_per_s()
            }
            ThroughputMode::Memory(_) => value.bytes_per_s(&key),
            ThroughputMode::Launch => 1.0 / value.duration_per_op().as_secs_f64(),
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

/// Where a declared resource's ceiling comes from.
///
/// Memory names only its direction: the working set is applied when the curve
/// is read, so two rows of different sizes share one sweep instead of each
/// asking the device for a probe of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Ceiling {
    Probe(ThroughputKey),
    Curve(MemoryAccess),
}

/// Which resource a bench row's binding measurement is judged against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    Compute,
    Read,
    Write,
    /// Getting the kernel onto the device at all, whatever it then does.
    Launch,
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

/// What a `(strategy, problem)` run honestly moves and computes, built from
/// `problem` alone and never from a measurement of its own run.
#[derive(Debug, Clone, Copy)]
pub struct CategoryWork {
    /// The compute the run performs, with the probe its ceiling comes from.
    /// `None` when the category has no honest count (an elementwise
    /// transcendental, an FFT); the adapter then declares no compute bound.
    pub compute: Option<ComputeWork>,
    /// Bytes read from global memory.
    pub bytes_read: usize,
    /// Bytes written to global memory.
    pub bytes_written: usize,
}

/// Operations a run performs, and the probe whose peak they are judged against.
///
/// The probe is declared because the two compute ceilings are different
/// hardware: an MMA kernel does not run where the scalar peak is measured.
#[derive(Debug, Clone, Copy)]
pub struct ComputeWork {
    /// Counted the way `key`'s probe counts them, so a multiply-add is 2 for
    /// [`ThroughputMode::ComputeDirect`].
    pub ops: usize,
    /// [`ThroughputMode::ComputeDirect`] for scalar arithmetic,
    /// [`ThroughputMode::ComputeCmma`] for a kernel on the tensor cores.
    pub key: ThroughputKey,
}

impl ComputeWork {
    /// Scalar arithmetic, the ceiling for a kernel that issues no MMA.
    pub fn direct(ops: usize, dtype: ElemType) -> Self {
        Self {
            ops,
            key: ThroughputKey {
                mode: ThroughputMode::ComputeDirect { dtype },
            },
        }
    }
}

impl CategoryWork {
    /// Launch joins the declared resources only under [`TimingMethod::System`];
    /// a device-timed row's timestamps begin once the dispatch is already paid
    /// for. It counts one, so a strategy launching several leaves it a floor.
    ///
    /// A memory probe is asked for the octave at or below the traffic, not the
    /// traffic itself: that is the grid [`MemoryCurve`] is measured on, so rows
    /// of neighbouring sizes share one probe instead of each paying for their
    /// own. The amount stays exact, since that is what the kernel moves.
    fn declared_resources(&self, timing: TimingMethod) -> Vec<(ResourceKind, usize, Ceiling)> {
        let mut out = Vec::with_capacity(4);

        if let Some(compute) = self.compute.filter(|c| c.ops > 0) {
            out.push((
                ResourceKind::Compute,
                compute.ops,
                Ceiling::Probe(compute.key),
            ));
        }
        if self.bytes_read > 0 {
            let read = Ceiling::Curve(MemoryAccess::Read);
            out.push((ResourceKind::Read, self.bytes_read, read));
        }
        if self.bytes_written > 0 {
            let write = Ceiling::Curve(MemoryAccess::Write);
            out.push((ResourceKind::Write, self.bytes_written, write));
        }
        if !out.is_empty() && timing == TimingMethod::System {
            let launch = Ceiling::Probe(ThroughputKey {
                mode: ThroughputMode::Launch,
            });
            out.push((ResourceKind::Launch, 1, launch));
        }

        out
    }

    /// The declared resources as `(kind, bound)` pairs, peaks pulled from the
    /// process-wide memo (see [`peak_per_s`]).
    fn resources(&self, timing: TimingMethod) -> Vec<(ResourceKind, ResourceBound)> {
        self.declared_resources(timing)
            .into_iter()
            .map(|(kind, amount, ceiling)| {
                let peak_per_s = match ceiling {
                    Ceiling::Probe(key) => peak_per_s(key),
                    Ceiling::Curve(access) => curve_ceiling(access, amount),
                };

                (kind, ResourceBound { amount, peak_per_s })
            })
            .collect()
    }
}

/// Scores `work` against measured device peaks, from the median sample
/// duration. Left unfilled when there are no samples or the median is zero.
fn score(mut samples: RunSamples, work: &CategoryWork, timing: TimingMethod) -> RunSamples {
    let Some(median_secs) = samples.median_secs() else {
        return samples;
    };

    let resources = work.resources(timing);
    let (tflops, binding) = score_bounds(median_secs, &resources);
    samples.tflops = tflops;
    samples.binding = binding;

    samples
}

/// The pure half of [`score`], so the selection is testable against fabricated
/// bounds rather than a device's measured peaks.
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

    /// Measures and memoizes every distinct peak the named problems declare, so
    /// the first timed row never pays for a probe. Returns how many distinct
    /// keys it warmed.
    fn warm_peaks(&self, problem_ids: &[String]) -> usize;

    fn run(
        &self,
        strategy_id: &str,
        problem_id: &str,
        num_samples: usize,
    ) -> Result<RunSamples, String>;

    /// `None` means the category doesn't expose a kernel result (e.g.
    /// memcpy_async: no semantic-level output).
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

    fn warm_peaks(&self, problem_ids: &[String]) -> usize {
        let mut seen = HashSet::new();
        for problem in Category::problems(self) {
            if !problem_ids.contains(&problem.id) {
                continue;
            }
            let Some(work) = Category::work(self, &problem.value) else {
                continue;
            };
            for (_, amount, ceiling) in work.declared_resources(Category::timing_method(self)) {
                if seen.insert(ceiling) {
                    match ceiling {
                        Ceiling::Probe(key) => peak_per_s(key),
                        Ceiling::Curve(access) => curve_ceiling(access, amount),
                    };
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
            Some(work) => score(samples, &work, Category::timing_method(self)),
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
            compute: (compute_ops > 0)
                .then(|| ComputeWork::direct(compute_ops, f32::elem_type_native())),
            bytes_read,
            bytes_written,
        }
    }

    /// The timing method that charges nothing for the dispatch.
    const NO_LAUNCH: TimingMethod = TimingMethod::Device;

    /// Zero-ops, zero-bytes work declares no resource, the launch it would
    /// still pay for included: the dispatch bound rides along with declared
    /// work rather than standing in for it.
    #[test]
    fn zero_work_declares_no_resources() {
        // Declares nothing, so this measures nothing.
        let resources = work(0, 0, 0).resources(TimingMethod::System);
        assert!(resources.is_empty());

        let (tflops, binding) = score_bounds(1.0, &resources);
        assert_eq!(tflops, None);
        assert!(binding.is_none());
    }

    /// A read-only declaration produces exactly one `Read` bound, carrying the
    /// declared byte count as its `amount`; no `Write` or `Compute` entry.
    #[test]
    fn read_only_work_declares_a_single_read_resource() {
        let resources = work(0, 4096, 0).declared_resources(NO_LAUNCH);
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].0, ResourceKind::Read);
        assert_eq!(resources[0].1, 4096);
    }

    /// A write-only declaration produces exactly one `Write` bound, mirroring
    /// the read-only case.
    #[test]
    fn write_only_work_declares_a_single_write_resource() {
        let resources = work(0, 0, 2048).declared_resources(NO_LAUNCH);
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].0, ResourceKind::Write);
        assert_eq!(resources[0].1, 2048);
    }

    /// A mixed declaration produces all three bounds, each carrying its own
    /// field's amount, compute first.
    #[test]
    fn mixed_work_declares_compute_read_and_write_resources() {
        let resources = work(1_000_000, 4096, 1024).declared_resources(NO_LAUNCH);
        assert_eq!(resources.len(), 3);
        assert_eq!(
            resources.iter().map(|(kind, ..)| *kind).collect::<Vec<_>>(),
            vec![
                ResourceKind::Compute,
                ResourceKind::Read,
                ResourceKind::Write
            ]
        );
        assert_eq!(resources[0].1, 1_000_000);
        assert_eq!(resources[1].1, 4096);
        assert_eq!(resources[2].1, 1024);
    }

    /// Every working set of one direction reads the same curve, so a sweep of
    /// problem shapes asks the device once per access rather than once per
    /// distinct size. The size is carried as the amount and applied when the
    /// curve is read.
    #[test]
    fn memory_resources_share_one_curve_per_access() {
        let small = work(0, 16384, 4096).declared_resources(NO_LAUNCH);
        let large = work(0, 1 << 30, 1 << 30).declared_resources(NO_LAUNCH);

        assert_eq!(small[0].2, Ceiling::Curve(MemoryAccess::Read));
        assert_eq!(small[1].2, Ceiling::Curve(MemoryAccess::Write));
        assert_eq!(small[0].2, large[0].2);
        assert_eq!(small[1].2, large[1].2);

        // Read and write are still distinct ceilings.
        assert_ne!(small[0].2, small[1].2);

        // The amount is exactly what the kernel moves.
        assert_eq!(small[0].1, 16384);
        assert_eq!(large[0].1, 1 << 30);
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

    /// `tflops` reads the compute entry whichever resource bound the run.
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

    /// A device-timed row's timestamps start once the dispatch is already
    /// paid for, so charging it for one would score it against time it never
    /// spent.
    #[test]
    fn launch_is_declared_only_when_the_timing_measures_it() {
        let timed = work(0, 0, 2048).declared_resources(TimingMethod::System);
        assert_eq!(timed.len(), 2);
        assert_eq!(timed[1].0, ResourceKind::Launch);
        assert_eq!(timed[1].1, 1);
        assert_eq!(
            timed[1].2,
            Ceiling::Probe(ThroughputKey {
                mode: ThroughputMode::Launch
            })
        );

        let untimed = work(0, 0, 2048).declared_resources(TimingMethod::Device);
        assert_eq!(untimed.len(), 1);
        assert_eq!(untimed[0].0, ResourceKind::Write);
    }

    /// A launch bound's fraction of peak is the share of the run the dispatch
    /// cost, not a rate: 8us of overhead in a 10us run reads 80%.
    #[test]
    fn score_bounds_binds_on_launch_for_a_run_that_barely_outlasts_it() {
        let write = ResourceBound {
            amount: 16_384,
            peak_per_s: 20e9,
        }; // 0.8us at peak, 8% of the run
        let launch = ResourceBound {
            amount: 1,
            peak_per_s: 125_000.0,
        }; // one launch every 8us

        let resources = vec![(ResourceKind::Write, write), (ResourceKind::Launch, launch)];
        let (_, binding) = score_bounds(10e-6, &resources);

        let binding = binding.expect("a usable peak binds");
        assert_eq!(binding.resource, ResourceKind::Launch);
        assert!((binding.fraction_of_peak - 0.8).abs() < 1e-9);
    }

    /// The same two resources on a run long enough to bury the dispatch.
    #[test]
    fn score_bounds_ignores_launch_once_the_run_outgrows_it() {
        let write = ResourceBound {
            amount: 2_000_000_000,
            peak_per_s: 20e9,
        }; // 0.1s at peak, the whole run
        let launch = ResourceBound {
            amount: 1,
            peak_per_s: 125_000.0,
        }; // 8us of a 0.1s run

        let resources = vec![(ResourceKind::Write, write), (ResourceKind::Launch, launch)];
        let (_, binding) = score_bounds(0.1, &resources);

        let binding = binding.expect("a usable peak binds");
        assert_eq!(binding.resource, ResourceKind::Write);
        assert_eq!(binding.fraction_of_peak, 1.0);
    }
}
