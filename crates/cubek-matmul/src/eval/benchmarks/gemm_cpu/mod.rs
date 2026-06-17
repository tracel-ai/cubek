//! Focused CPU GEMM comparison: CpuGemm

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::eval::benchmarks::gemm::{self, GemmProblem};
use crate::strategy::Strategy;

/// CpuGemm vs the simple-unit baseline, plus the forced-tile mask probes. The forced
/// tiles are diagnostic on the 512 square: `t64`/`t32` divide 512 (maskless), `t48`
/// does not (masked).
const STRATEGIES: &[&str] = &["simple_unit_min", "simple_unit_max", "cpu_gemm"];

/// Base shapes; the catalog expands each over all four layouts (rr/rc/cr/cc) and both
/// precisions (f32/f16). The 512 square keeps the CPU reference cheap while still
/// exercising the vectorized (row-major) and scalar (col-major) paths under masking.
const SHAPES: &[&str] = &["rect_1x512x512x512"];

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = GemmProblem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "gemm_cpu"
    }

    fn label(&self) -> &'static str {
        "GEMM (CPU)"
    }

    fn problems(&self) -> Vec<CatalogEntry<GemmProblem>> {
        gemm::problems()
            .into_iter()
            .filter(|p| SHAPES.iter().any(|s| p.id.starts_with(&format!("{s}_"))))
            .collect()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        gemm::strategies()
            .into_iter()
            .filter(|s| STRATEGIES.contains(&s.id.as_str()))
            .collect()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &GemmProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        gemm::bench(strategy, problem, num_samples)
    }

    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = GemmProblem, Strategy = Strategy>>
    {
        Some(&gemm::GemmCorrectness)
    }
}
