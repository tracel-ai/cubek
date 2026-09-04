use cubek_test_utils::{HostData, Progress, ValidationResult, assert_equals_approx};

use crate::ReduceStrategy;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};
use crate::eval::cpu_reference::{
    RAMP_MAX_ELEMS, ReduceInput, ReduceValues, comparison_epsilon, cpu_reference_result,
    strategy_result, strategy_result_with_indices,
};

pub struct ReduceCorrectness;

impl ReduceCorrectness {
    /// A strategy that computes the wrong answer would still time fast, so every
    /// strategy proves itself on a small shape before it is measured.
    pub fn verify(strategy: &ReduceStrategy, problem: &ReduceProblem) -> Result<(), String> {
        let proof = ReduceProblem {
            shape: proof_shape(problem.shape.len(), problem.axis),
            axis: problem.axis,
            config: problem.config,
            kind: problem.kind,
            precision: problem.precision,
        };
        let input = ReduceInput {
            dtype: proof.precision.dtype(),
            values: ReduceValues::Ramp,
        };

        let actual = Self::kernel_output(strategy, &proof, input)?;
        let expected = Self::reference_output(&proof, input, None)?;

        match assert_equals_approx(&actual, &expected, comparison_epsilon(proof.config)) {
            ValidationResult::Pass | ValidationResult::Skipped(_) => Ok(()),
            ValidationResult::Fail(reason) | ValidationResult::Error(reason) => Err(format!(
                "{:?} computes the wrong {:?} at {:?}, so its timing would be meaningless: {reason}",
                strategy, proof.config, proof.shape
            )),
        }
    }

    /// Validate the path that is actually benchmarked. The two-launch kind runs
    /// the same `reduce` as `Single` for its values half, so only the fused kind
    /// needs the dedicated entrypoint.
    fn kernel_output(
        strategy: &ReduceStrategy,
        problem: &ReduceProblem,
        input: ReduceInput,
    ) -> Result<HostData, String> {
        let client = cubecl::test_device().client();
        match problem.kind {
            ReduceBenchKind::Single | ReduceBenchKind::TwoLaunch => strategy_result(
                client,
                problem.shape.clone(),
                problem.axis,
                strategy.clone(),
                problem.config,
                input,
            ),
            ReduceBenchKind::Fused => strategy_result_with_indices(
                client,
                problem.shape.clone(),
                problem.axis,
                strategy.clone(),
                problem.config,
                input,
            ),
        }
    }

    fn reference_output(
        problem: &ReduceProblem,
        input: ReduceInput,
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let client = cubecl::test_device().client();
        cpu_reference_result(
            client,
            problem.shape.clone(),
            problem.axis,
            problem.config,
            input,
            progress,
        )
    }
}

/// The shape a strategy proves itself on: [`RAMP_MAX_ELEMS`] split between the
/// axes, all of it on the reduced one, so that axis stays long enough for a
/// plane or cube routine to actually fold.
fn proof_shape(rank: usize, axis: usize) -> Vec<usize> {
    let mut shape = vec![2; rank];
    shape[axis] = RAMP_MAX_ELEMS >> (rank - 1);
    shape
}

impl cubek_test_utils::Correctness for ReduceCorrectness {
    type Problem = ReduceProblem;
    type Strategy = ReduceStrategy;

    fn kernel_result(
        &self,
        strategy: &ReduceStrategy,
        problem: &ReduceProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        Self::kernel_output(
            strategy,
            problem,
            ReduceInput::uniform(problem.precision.dtype(), seeds[0]),
        )
    }

    fn reference_result(
        &self,
        problem: &ReduceProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        Self::reference_output(
            problem,
            ReduceInput::uniform(problem.precision.dtype(), seeds[0]),
            progress,
        )
    }
}
