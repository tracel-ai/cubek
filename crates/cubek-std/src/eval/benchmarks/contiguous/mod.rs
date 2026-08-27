mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{ContiguousProblem, problems};
pub use strategy::{ContiguousStrategy, strategies};

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = ContiguousProblem;
    type Strategy = ContiguousStrategy;

    fn id(&self) -> &'static str {
        "contiguous"
    }

    fn label(&self) -> &'static str {
        "Contiguous"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<ContiguousProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<ContiguousStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &ContiguousStrategy,
        problem: &ContiguousProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    /// A layout copy: reads the strided input and writes a fresh contiguous
    /// buffer of the same element count. No compute beyond addressing.
    fn work(&self, problem: &ContiguousProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elems = problem.shape.iter().product::<usize>() * dtype.size();

        Some(CategoryWork {
            compute: None,
            bytes_read: elems,
            bytes_written: elems,
        })
    }
}
