use cubecl::client::Client;
use cubek_test_utils::{HostData, Progress, ValidationResult, assert_equals_approx};

use crate::ReduceStrategy;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};
use crate::eval::cpu_reference::{
    RAMP_MAX_ELEMS, ReduceInput, ReduceValues, comparison_epsilon, cpu_reference_result,
    strategy_result, strategy_result_with_indices,
};

pub struct ReduceCorrectness;

impl ReduceCorrectness {
    /// Proves `strategy` correct on a small shape before it's timed, so a wrong
    /// kernel can't win a benchmark by being fast.
    pub fn verify(strategy: &ReduceStrategy, problem: &ReduceProblem) -> Result<(), String> {
        let client = cubecl::test_device().client();
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

        let actual = Self::kernel_output(client.clone(), strategy, &proof, input)?;
        let expected = Self::reference_output(client, &proof, input, None)?;

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
        client: Client,
        strategy: &ReduceStrategy,
        problem: &ReduceProblem,
        input: ReduceInput,
    ) -> Result<HostData, String> {
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
        client: Client,
        problem: &ReduceProblem,
        input: ReduceInput,
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
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

/// At most [`RAMP_MAX_ELEMS`] elements, all on the reduced axis so a plane or
/// cube routine has enough to fold. Padding other axes with 2 keeps the total
/// a power of two at any rank, instead of the reduced axis underflowing to
/// zero once rank exceeds `RAMP_MAX_ELEMS`'s bit width.
fn proof_shape(rank: usize, axis: usize) -> Vec<usize> {
    let mut shape = vec![1; rank];
    let mut budget = RAMP_MAX_ELEMS;
    for (i, dim) in shape.iter_mut().enumerate() {
        if i == axis || budget == 1 {
            continue;
        }
        *dim = 2;
        budget /= 2;
    }
    shape[axis] = budget;
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
        let client = cubecl::test_device().client();
        Self::kernel_output(
            client,
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
        let client = cubecl::test_device().client();
        Self::reference_output(
            client,
            problem,
            ReduceInput::uniform(problem.precision.dtype(), seeds[0]),
            progress,
        )
    }
}

#[cfg(test)]
mod proof_shape_tests {
    use super::proof_shape;
    use crate::eval::cpu_reference::RAMP_MAX_ELEMS;

    #[test]
    fn stays_within_ramp_bounds_at_every_rank() {
        for rank in 1..=16 {
            for axis in 0..rank {
                let shape = proof_shape(rank, axis);
                let elems: usize = shape.iter().product();
                assert!(
                    elems.is_power_of_two() && elems <= RAMP_MAX_ELEMS,
                    "rank {rank} axis {axis}: {shape:?} has {elems} elements"
                );
            }
        }
    }

    #[test]
    fn matches_ramp_max_elems_below_its_bit_width() {
        assert_eq!(proof_shape(3, 2), vec![2, 2, 512]);
    }
}
