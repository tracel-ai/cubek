use cubecl::client::Client;
use cubek_test_utils::{HostData, Progress, ValidationResult, assert_equals_approx};

use crate::eval::benchmarks::benchmark::two_launch_operations;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};
use crate::eval::benchmarks::results::{
    RAMP_MAX_ELEM_COUNT, ReduceInput, ReduceRunError, ReduceValues, cpu_reference_result,
    strategy_result,
};
use crate::kernels::ReduceOpStrategy;

/// Proves and checks the reductions the catalogue times, against the CPU reference.
pub(crate) struct ReduceVerifier;

impl ReduceVerifier {
    /// Proves `strategy` correct on a small shape before it's timed, so a wrong
    /// kernel can't win a benchmark by being fast.
    ///
    /// # Errors
    ///
    /// Where its result differs from the reference's, or it does not run the shape.
    pub fn verify(strategy: &ReduceOpStrategy, problem: &ReduceProblem) -> Result<(), String> {
        let client = cubecl::test_device().client();
        let proof = ReduceProblem {
            shape: proof_shape(&problem.shape, problem.axis),
            axis: problem.axis,
            operation: problem.operation,
            kind: problem.kind,
            precision: problem.precision,
        };
        let input = ReduceInput {
            dtype: proof.precision.to_dtype(),
            values: ReduceValues::Ramp,
        };

        let actual = Self::kernel_output(client.clone(), strategy, &proof, input)
            .map_err(|why| why.to_string())?;
        let expected = Self::reference_output(client, &proof, input, None)?;

        match assert_equals_approx(&actual, &expected, proof.tolerance()) {
            ValidationResult::Pass | ValidationResult::Skipped(_) => Ok(()),
            ValidationResult::Fail(reason) | ValidationResult::Error(reason) => Err(format!(
                "{:?} computes the wrong {:?} at {:?}, so its timing would be meaningless: {reason}",
                strategy, proof.operation, proof.shape
            )),
        }
    }
}

impl ReduceVerifier {
    /// Validate the path that is actually benchmarked. The two-launch kind runs
    /// the same `Reduce::execute` as `Single` for its values half, so only the fused kind
    /// needs the dedicated entrypoint.
    fn kernel_output(
        client: Client,
        strategy: &ReduceOpStrategy,
        problem: &ReduceProblem,
        input: ReduceInput,
    ) -> Result<HostData, ReduceRunError> {
        strategy_result(
            client,
            problem.shape.clone(),
            problem.axis,
            *strategy,
            problem.operation,
            input,
            problem.kind == ReduceBenchKind::Fused,
        )
    }

    /// What [`kernel_output`](Self::kernel_output) should hold: the values half of a fused launch,
    /// what the operation writes otherwise.
    fn reference_output(
        client: Client,
        problem: &ReduceProblem,
        input: ReduceInput,
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let operation = match problem.kind {
            ReduceBenchKind::Single | ReduceBenchKind::TwoLaunch => problem.operation,
            ReduceBenchKind::Fused => two_launch_operations(problem.operation).0,
        };
        cpu_reference_result(
            client,
            problem.shape.clone(),
            problem.axis,
            operation,
            input,
            progress,
        )
    }
}

/// At most [`RAMP_MAX_ELEM_COUNT`] elements, all on the reduced axis so a plane or
/// cube routine has enough to reduce. Padding other axes with 2 keeps the total
/// a power of two at any rank, instead of the reduced axis underflowing to
/// zero once rank exceeds `RAMP_MAX_ELEM_COUNT`'s bit width. An axis of one in
/// `shape` stays one, so a whole tensor's one row is proved as one row.
fn proof_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    let mut proof = vec![1; shape.len()];
    let mut budget = RAMP_MAX_ELEM_COUNT;
    for (i, dim) in proof.iter_mut().enumerate() {
        if i == axis || budget == 1 || shape[i] == 1 {
            continue;
        }
        *dim = 2;
        budget /= 2;
    }
    proof[axis] = budget;
    proof
}

impl cubek_test_utils::Correctness for ReduceVerifier {
    type Problem = ReduceProblem;
    type Strategy = ReduceOpStrategy;

    fn kernel_result(
        &self,
        strategy: &ReduceOpStrategy,
        problem: &ReduceProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let client = cubecl::test_device().client();
        Self::kernel_output(client, strategy, problem, seeded_input(problem, seeds))
            .map_err(|why| why.to_string())
    }

    fn reference_result(
        &self,
        problem: &ReduceProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let client = cubecl::test_device().client();
        Self::reference_output(client, problem, seeded_input(problem, seeds), progress)
    }
}

/// The input `problem` reduces, drawn uniformly in its precision from the first of `seeds`.
fn seeded_input(problem: &ReduceProblem, seeds: &[u64]) -> ReduceInput {
    ReduceInput::uniform(problem.precision.to_dtype(), seeds[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReduceOperation;
    use crate::eval::benchmarks::problem::ReduceBenchPrecision;
    use crate::kernels::ReduceKernel;
    use cubek_test_utils::Correctness;

    /// A proof shape whose values a ramp permutes: a power of two at most the ramp's longest.
    fn fits_a_ramp(proof: &[usize]) -> bool {
        let elems: usize = proof.iter().product();
        elems.is_power_of_two() && elems <= RAMP_MAX_ELEM_COUNT
    }

    /// The fewest axes keep within a ramp.
    #[test]
    fn one_axis_stays_within_ramp_bounds() {
        assert!(fits_a_ramp(&proof_shape(&[3], 0)));
    }

    /// The most axes, reduced along the first, keep within a ramp.
    #[test]
    fn the_most_axes_along_the_first_stay_within_ramp_bounds() {
        const AXIS_COUNT: usize = 16;
        assert!(fits_a_ramp(&proof_shape(&[3; AXIS_COUNT], 0)));
    }

    /// The most axes, reduced along the last, keep within a ramp.
    #[test]
    fn the_most_axes_along_the_last_stay_within_ramp_bounds() {
        const AXIS_COUNT: usize = 16;
        assert!(fits_a_ramp(&proof_shape(&[3; AXIS_COUNT], AXIS_COUNT - 1)));
    }

    #[test]
    fn matches_ramp_max_elems_below_its_bit_width() {
        assert_eq!(proof_shape(&[3, 3, 3], 2), vec![2, 2, 512]);
    }

    #[test]
    fn keeps_a_whole_tensor_one_row() {
        assert_eq!(proof_shape(&[1, 16_777_216], 1), vec![1, 2048]);
    }

    const UNIT: ReduceOpStrategy = ReduceOpStrategy::Specific(ReduceKernel::Unit);

    /// A reduction whose reference is cheap, launched as `kind` says.
    fn small_problem(kind: ReduceBenchKind) -> ReduceProblem {
        ReduceProblem {
            shape: vec![4, 300],
            axis: 1,
            operation: ReduceOperation::Max,
            kind,
            precision: ReduceBenchPrecision::F32,
        }
    }

    #[test]
    fn a_right_kernel_is_proved() {
        let proved = ReduceVerifier::verify(&UNIT, &small_problem(ReduceBenchKind::Single));
        assert_eq!(proved, Ok(()));
    }

    /// The kernel and the reference draw one input from the seed.
    #[test]
    fn a_fused_kernels_values_are_the_references() {
        let problem = small_problem(ReduceBenchKind::Fused);
        let written = ReduceVerifier
            .kernel_result(&UNIT, &problem, &[7])
            .expect("the unit kernel serves every problem");
        let expected = ReduceVerifier
            .reference_result(&problem, &[7], None)
            .expect("the reference reduces every problem");
        assert_equals_approx(&written, &expected, problem.tolerance())
            .as_test_outcome()
            .enforce();
    }

    /// Correctness over the benchmark catalogue: each pair checked at both precisions, under a
    /// module named for the shape so a second one does not collide. The list is a subset because
    /// one pair folds 67M elements against the host reference; what is missing has no check at all.
    mod catalog {
        use super::*;
        use crate::eval::benchmarks::{ReduceBenchPrecision, ReduceCategory};
        use cubek_test_utils::{CatalogEntry, Category as _, TestOutcome, ValidationResult};

        const SEEDS: [u64; 2] = [12, 34];

        fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
            entries
                .into_iter()
                .find(|e| e.id == id)
                .unwrap_or_else(|| panic!("unknown id: {id}"))
                .value
        }

        /// Checks the strategy `strategy_id` on the problem `problem_id` against the reference: a
        /// decline is a skip, a [`Decline::Broken`](cubecl::tile::launch::Decline::Broken) a
        /// failure.
        fn check(strategy_id: &str, problem_id: &str) {
            let strategy: ReduceOpStrategy = lookup(ReduceCategory.strategies(), strategy_id);
            let problem: ReduceProblem = lookup(ReduceCategory.problems(), problem_id);
            let client = cubecl::test_device().client();
            let input = seeded_input(&problem, &SEEDS);
            let actual = match ReduceVerifier::kernel_output(client, &strategy, &problem, input) {
                Ok(host) => host,
                Err(ReduceRunError::Declined(why)) => {
                    return TestOutcome::refused(&problem_id, &why).enforce();
                }
                Err(ReduceRunError::OutsideDomain(why)) => {
                    panic!("{problem_id} lies outside the reduction's Domain: {why}")
                }
                Err(ReduceRunError::DidNotRun(why)) => {
                    return TestOutcome::CompileError(why).enforce();
                }
            };
            let expected = ReduceVerifier
                .reference_result(&problem, &SEEDS, None)
                .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));
            assert_equals_approx(&actual, &expected, problem.tolerance())
                .as_test_outcome()
                .enforce();
        }

        /// The check of the problem's f16 row, which the catalogue offers only where the device
        /// folds f16, so there is nothing to look up otherwise: a skip.
        fn check_f16(strategy_id: &str, problem_id: &str) {
            let f16 = ReduceBenchPrecision::F16;
            match f16.is_offered_on(&cubecl::test_device().client()) {
                true => check(strategy_id, &format!("{problem_id}{}", f16.suffix())),
                false => TestOutcome::Validated(ValidationResult::Skipped(
                    "no f16 row on this device".to_string(),
                ))
                .enforce(),
            }
        }

        macro_rules! bench_catalog {
            ($shape:ident; $($problem:ident: [$($strategy:ident),+ $(,)?]),+ $(,)?) => {
                mod $shape {
                    use super::*;

                    $(mod $problem {
                        use super::*;

                        $(mod $strategy {
                            use super::*;

                            const STRATEGY: &str = stringify!($strategy);
                            const PROBLEM: &str =
                                concat!(stringify!($problem), "_", stringify!($shape));

                            #[test]
                            fn f32() {
                                check(STRATEGY, PROBLEM);
                            }

                            #[test]
                            fn f16() {
                                check_f16(STRATEGY, PROBLEM);
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
    }
}
