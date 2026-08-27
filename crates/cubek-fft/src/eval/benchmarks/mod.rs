//! Benchmark catalogue for `cubek-fft`.

mod benchmark;
mod correctness;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use correctness::FftCorrectness;
pub use problem::{FftProblem, problems};
pub use strategy::{FftStrategy, strategies};

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

use crate::FftMode;

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = FftProblem;
    type Strategy = FftStrategy;

    fn id(&self) -> &'static str {
        "fft"
    }

    fn label(&self) -> &'static str {
        "FFT"
    }

    fn problems(&self) -> Vec<CatalogEntry<FftProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<FftStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &FftStrategy,
        problem: &FftProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = FftProblem, Strategy = FftStrategy>>
    {
        Some(&FftCorrectness)
    }

    /// No honest op count across algorithms/radices, so this declares reads and
    /// writes only. The real (`Forward`) side is the full signal length; the
    /// spectrum side is the one-sided `len/2 + 1` complex layout, so its two planes
    /// (real, imaginary) together move less than a same-length real buffer.
    fn work(&self, problem: &FftProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elem_size = dtype.size();

        let signal_elems: usize = problem.shape.iter().product();
        let last = problem.shape.len() - 1;
        let spectrum_elems: usize =
            signal_elems / problem.shape[last] * (problem.shape[last] / 2 + 1);

        let (read_elems, written_elems) = match problem.mode {
            FftMode::Forward => (signal_elems, 2 * spectrum_elems),
            FftMode::Inverse => (2 * spectrum_elems, signal_elems),
        };

        Some(CategoryWork {
            compute: None,
            bytes_read: read_elems * elem_size,
            bytes_written: written_elems * elem_size,
        })
    }
}
