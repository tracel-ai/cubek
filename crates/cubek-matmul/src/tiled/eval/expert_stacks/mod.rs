//! The routed and grouped products against an expert stack laid three ways: rows padded as CUDA's
//! and HIP's pitched allocator pads them, the same rows dense, and tile order. The stacks are
//! Qwen3.8-Flash-Next's, whose rows the pitch pads by 24 % (`gate_up`) and 60 % (`down`); the
//! question is what each layout costs a decode and a prefill.
//!
//! Each call is the product a model's layer makes: the stack restated as its slabs, the tensors
//! built, the autotune's winner launched. A decode reads another set of experts each call, so its
//! slabs come from memory rather than the cache.

mod benchmark;
mod problem;
mod strategy;

pub use benchmark::{bench, interleaved};
pub use problem::{ExpertProduct, ExpertStackProblem, problems};
pub use strategy::{StackLayout, strategies};

use cubek_test_utils::{CatalogEntry, CategoryWork, RunSamples};

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = ExpertStackProblem;
    type Strategy = StackLayout;

    fn id(&self) -> &'static str {
        "expert_stacks"
    }

    fn label(&self) -> &'static str {
        "Expert stacks: padded rows, dense rows, tile order"
    }

    fn timing_method(&self) -> cubecl::benchmark::TimingMethod {
        cubek_test_utils::timing_method(cubecl::benchmark::TimingMethod::Device)
    }

    fn problems(&self) -> Vec<CatalogEntry<ExpertStackProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<StackLayout>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &StackLayout,
        problem: &ExpertStackProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    /// The bytes the product needs, whatever the layout: each slab it reads, `n` rows of `k/2`
    /// bytes of codes and `k/16` of scales, and the activation; the `f16` output. Padding and
    /// tile order's stagger are not counted, so a layout that reads them shows as slower against
    /// the same bytes.
    fn work(&self, problem: &ExpertStackProblem) -> Option<CategoryWork> {
        let slab = problem.n * (problem.k / 2 + problem.k / 16);
        let lhs = problem.lhs_rows() * problem.k * 2;
        let out_rows = match problem.product {
            ExpertProduct::Routed { tokens } => tokens * problem.picks,
            ExpertProduct::Grouped { .. } => problem.lhs_rows(),
        };
        Some(CategoryWork {
            compute: None,
            bytes_read: problem.slabs_read() * slab + lhs,
            bytes_written: out_rows * problem.n * 2,
        })
    }
}
