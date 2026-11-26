use std::collections::HashSet;

use cubecl::benchmark::TimingMethod;

use super::base::{Ceiling, curve_ceiling, peak_per_s, score};
use crate::{
    CatalogEntry, Category, HostData, ItemDescriptor, KernelResultError, Progress, RunSamples,
};

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
    ///
    /// # Errors
    ///
    /// As [`Correctness::kernel_result`](crate::Correctness::kernel_result), and
    /// [`KernelResultError::NotRun`] where the category lists no such strategy or problem.
    fn kernel_result(
        &self,
        _strategy_id: &str,
        _problem_id: &str,
        _seed_lhs: u64,
        _seed_rhs: u64,
    ) -> Option<Result<HostData, KernelResultError>> {
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
    ) -> Option<Result<HostData, KernelResultError>> {
        let correctness = Category::correctness(self)?;
        let problems = Category::problems(self);
        let problem = match problems.iter().find(|e| e.id == problem_id) {
            Some(p) => p,
            None => return Some(Err(format!("unknown problem: {problem_id}").into())),
        };
        let strategies = Category::strategies(self);
        let strategy = match strategies.iter().find(|e| e.id == strategy_id) {
            Some(s) => s,
            None => return Some(Err(format!("unknown strategy: {strategy_id}").into())),
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
