mod benchmark;
mod problem;
mod strategy;

pub use benchmark::bench;
pub use problem::{MemcpyAsyncProblem, problems};
pub use strategy::{CopyStrategyEnum, strategies};

use cubecl::prelude::*;
use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = MemcpyAsyncProblem;
    type Strategy = CopyStrategyEnum;

    fn id(&self) -> &'static str {
        "memcpy_async"
    }

    fn label(&self) -> &'static str {
        "Memcpy (async)"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubecl::benchmark::TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<MemcpyAsyncProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<CopyStrategyEnum>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &CopyStrategyEnum,
        problem: &MemcpyAsyncProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    /// Copy-shaped: the kernel streams `data_count` elements through shared memory
    /// and folds them into a `window_size`-sized accumulator that becomes the
    /// output, so the global write is the small window, not the full stream.
    fn work(&self, problem: &MemcpyAsyncProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let elem_size = dtype.size();

        Some(CategoryWork {
            compute: None,
            bytes_read: problem.data_count * elem_size,
            bytes_written: problem.window_size * elem_size,
        })
    }
}
