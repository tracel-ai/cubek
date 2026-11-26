//! Correctness over the reduce benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_reduce::eval::benchmarks::{
    BROKEN_LAUNCH, ReduceBenchPrecision, ReduceCategory, ReduceProblem,
};
use cubek_reduce::kernels::ReduceOpStrategy;
use cubek_test_utils::{
    CatalogEntry, Category as _, TestOutcome, ValidationResult, assert_equals_approx,
};

const SEEDS: [u64; 2] = [12, 34];

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

/// Checks the strategy of one id on the problem of another, against the reference.
struct CatalogChecker {
    strategy_id: &'static str,
    problem_id: &'static str,
}

impl CatalogChecker {
    pub fn new(strategy_id: &'static str, problem_id: &'static str) -> Self {
        Self {
            strategy_id,
            problem_id,
        }
    }

    pub fn check(&self) {
        self.check_problem(self.problem_id);
    }

    /// The check of the problem's f16 row, which the catalogue offers only where the device folds
    /// f16, so there is nothing to look up otherwise: a skip.
    pub fn check_f16(&self) {
        let f16 = ReduceBenchPrecision::F16;
        match f16.is_offered_on(&cubecl::test_device().client()) {
            true => self.check_problem(&format!("{}{}", self.problem_id, f16.suffix())),
            false => TestOutcome::Validated(ValidationResult::Skipped(
                "no f16 row on this device".to_string(),
            ))
            .enforce(),
        }
    }
}

impl CatalogChecker {
    fn check_problem(&self, problem_id: &str) {
        let strategy: ReduceOpStrategy = lookup(ReduceCategory.strategies(), self.strategy_id);
        let problem: ReduceProblem = lookup(ReduceCategory.problems(), problem_id);

        let correctness = ReduceCategory
            .correctness()
            .expect("the reduce catalogue checks its problems");
        let actual = match correctness.kernel_result(&strategy, &problem, &SEEDS) {
            Ok(host) => host,
            Err(broken) if broken.contains(BROKEN_LAUNCH) => panic!("{problem_id}: {broken}"),
            Err(e) => return TestOutcome::CompileError(e).enforce(),
        };
        let expected = correctness
            .reference_result(&problem, &SEEDS, None)
            .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

        assert_equals_approx(&actual, &expected, problem.tolerance())
            .as_test_outcome()
            .enforce();
    }
}

/// Each pair is checked at both precisions, under a module named for the shape
/// so a second one does not collide. The list is a subset because one pair folds
/// 67M elements against the host reference; what is missing has no check at all.
macro_rules! bench_catalog {
    ($shape:ident; $($problem:ident: [$($strategy:ident),+ $(,)?]),+ $(,)?) => {
        mod $shape {
            use super::*;

            $(mod $problem {
                use super::*;

                $(mod $strategy {
                    use super::*;

                    const STRATEGY: &str = stringify!($strategy);
                    const PROBLEM: &str = concat!(stringify!($problem), "_", stringify!($shape));

                    #[test]
                    fn f32() {
                        CatalogChecker::new(STRATEGY, PROBLEM).check();
                    }

                    #[test]
                    fn f16() {
                        CatalogChecker::new(STRATEGY, PROBLEM).check_f16();
                    }
                })+
            })+
        }
    };
}

bench_catalog! {
    axis2_32x512x4095;

    sum: [unit, plane_shuffle],
    arg_topk1: [unit],
    arg_topk2: [unit],
    arg_topk3: [unit],

    // The fused entries get timed against the two-launch ones, so they have to be
    // correct at the benchmark's own shape and strategy, not only at the small
    // shapes the integration tests cover.
    topk2_fused: [cube_staged],
    topk3_fused: [unit, cube_staged],
    topk3_two_launch: [cube_staged],
    max_fused: [unit, cube_staged],
    min_fused: [unit, cube_staged],
}

bench_catalog! {
    sampler_1x151936;

    // The rule splits this one long row across cubes, on the hypercube.
    arg_topk1: [tile],
    topk20: [tile],
}
