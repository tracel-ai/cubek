//! Correctness over the reduce benchmark catalogue.

#![cfg(feature = "benchmarks")]

use cubek_reduce::ReduceStrategy;
use cubek_reduce::eval::benchmarks::{
    ReduceBenchPrecision, ReduceCorrectness, ReduceProblem, precisions, problems, strategies,
};
use cubek_reduce::eval::cpu_reference::comparison_epsilon;
use cubek_test_utils::{CatalogEntry, Correctness, TestOutcome, assert_equals_approx};

const SEEDS: [u64; 2] = [12, 34];

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run(strategy_id: &str, problem_id: &str) {
    let strategy: ReduceStrategy = lookup(strategies(), strategy_id);
    let problem: ReduceProblem = lookup(problems(), problem_id);

    let actual = match ReduceCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = ReduceCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, comparison_epsilon(problem.config))
        .as_test_outcome()
        .enforce();
}

/// Takes the f32 id and derives the f16 one, since the catalogue offers f16
/// rows only where the device can fold them and there is nothing to look up
/// otherwise.
fn run_f16(strategy_id: &str, problem_id: &str) {
    let f16 = ReduceBenchPrecision::F16;
    if !precisions().contains(&f16) {
        return;
    }
    run(strategy_id, &format!("{problem_id}{}", f16.suffix()));
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
                        run(STRATEGY, PROBLEM);
                    }

                    #[test]
                    fn f16() {
                        run_f16(STRATEGY, PROBLEM);
                    }
                })+
            })+
        }
    };
}

bench_catalog! {
    axis2_32x512x4095;

    sum: [unit_parallel, plane_parallel],
    arg_topk1: [unit_parallel],
    arg_topk2: [unit_parallel],
    arg_topk3: [unit_parallel],

    // The fused entries get timed against the two-launch ones, so they have to be
    // correct at the benchmark's own shape and strategy, not only at the small
    // shapes the integration tests cover.
    topk2_fused: [cube_serial],
    topk3_fused: [unit_parallel, cube_serial],
    topk3_two_launch: [cube_serial],
    max_fused: [unit_parallel, cube_serial],
    min_fused: [unit_parallel, cube_serial],
}
