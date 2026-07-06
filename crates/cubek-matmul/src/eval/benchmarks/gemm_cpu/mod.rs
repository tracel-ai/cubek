//! Focused CPU GEMM comparison: CpuGemm

use cubek_test_utils::{CatalogEntry, RunSamples};

use crate::eval::benchmarks::gemm::{self, GemmProblem};
use crate::strategy::Strategy;

/// CpuGemm (auto plane grid) against the simple-unit baseline, plus the fast-core scaling
/// probes (`fast_p1`→`fast_p16`) that fix the register-fit leaf (no spill) and fan out
/// 1→16 worker threads — the multi-core study this category exists for: how the *fast*
/// single-core instruction spreads across threads.
const STRATEGIES: &[&str] = &[
    "simple_unit_max",
    "cpu_gemm",
    "cpu_gemm_fast_p1",
    "cpu_gemm_fast_p4",
    "cpu_gemm_fast_p8",
    "cpu_gemm_fast_p16",
];

/// Base shapes, restricted below to the row/row layout — the only layout CpuGemm
/// vectorizes (rhs and output both N-contiguous), so the only one whose throughput is
/// representative. The 512 square keeps the CPU reference cheap; `square_2x1024` gives the
/// plane fan-out enough work to scale past launch overhead; `square_1x1536` is non-power-of-two
/// so its plane grid divides evenly across a 12-core machine (the power-of-two squares can't).
const SHAPES: &[&str] = &["rect_1x512x512x512", "square_2x1024", "square_1x1536"];

/// The optimal (vectorized) layout: row-major lhs *and* rhs. The catalog encodes it as the
/// `_rr_` segment of a problem id.
const LAYOUT: &str = "_rr_";

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
            .filter(|p| p.id.contains(LAYOUT))
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
