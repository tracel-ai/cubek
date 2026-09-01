mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{UnaryProblem, problems};
pub use strategy::{UnaryStrategy, strategies};

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = UnaryProblem;
    type Strategy = UnaryStrategy;

    fn id(&self) -> &'static str {
        "unary"
    }

    fn label(&self) -> &'static str {
        "Unary"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<UnaryProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<UnaryStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &UnaryStrategy,
        problem: &UnaryProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    /// No honest op count: the kernel's cost is 256 rounds of `cos`, and a
    /// transcendental has no fixed multiply-add price to compare against
    /// `ComputeDirect`. Reads and writes are the elementwise shape: two
    /// operands in, one out.
    fn work(&self, problem: &UnaryProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elems = problem.shape.iter().product::<usize>() * dtype.size();

        Some(CategoryWork {
            compute: None,
            bytes_read: 2 * elems,
            bytes_written: elems,
        })
    }
}
