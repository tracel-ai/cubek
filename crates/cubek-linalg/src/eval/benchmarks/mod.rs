//! Benchmark catalogue for `cubek-linalg`.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{QrProblem, problems};
pub use strategy::{QrStrategy, strategies};

use cubek_test_utils::{CatalogEntry, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = QrProblem;
    type Strategy = QrStrategy;

    fn id(&self) -> &'static str {
        "qr"
    }

    fn label(&self) -> &'static str {
        "QR decomposition"
    }

    fn problems(&self) -> Vec<CatalogEntry<QrProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<QrStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &QrStrategy,
        problem: &QrProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
