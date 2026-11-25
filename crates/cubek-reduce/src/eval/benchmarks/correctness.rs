use cubecl::client::Client;
use cubek_test_utils::{HostData, Progress, ValidationResult, assert_equals_approx};

use crate::eval::benchmarks::benchmark::two_launch_operations;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};
use crate::eval::cpu_reference::{
    RAMP_MAX_ELEMS, ReduceInput, ReduceValues, comparison_epsilon, cpu_reference_result,
    strategy_result, strategy_result_with_indices,
};
use crate::kernels::ReduceOpStrategy;

pub struct ReduceCorrectness;

impl ReduceCorrectness {
    /// Proves `strategy` correct on a small shape before it's timed, so a wrong
    /// kernel can't win a benchmark by being fast.
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

        let actual = Self::kernel_output(client.clone(), strategy, &proof, input)?;
        let expected = Self::reference_output(client, &proof, input, None)?;

        match assert_equals_approx(&actual, &expected, comparison_epsilon(proof.operation)) {
            ValidationResult::Pass | ValidationResult::Skipped(_) => Ok(()),
            ValidationResult::Fail(reason) | ValidationResult::Error(reason) => Err(format!(
                "{:?} computes the wrong {:?} at {:?}, so its timing would be meaningless: {reason}",
                strategy, proof.operation, proof.shape
            )),
        }
    }
}

impl ReduceCorrectness {
    /// Validate the path that is actually benchmarked. The two-launch kind runs
    /// the same `Reduce::execute` as `Single` for its values half, so only the fused kind
    /// needs the dedicated entrypoint.
    fn kernel_output(
        client: Client,
        strategy: &ReduceOpStrategy,
        problem: &ReduceProblem,
        input: ReduceInput,
    ) -> Result<HostData, String> {
        match problem.kind {
            ReduceBenchKind::Single | ReduceBenchKind::TwoLaunch => strategy_result(
                client,
                problem.shape.clone(),
                problem.axis,
                *strategy,
                problem.operation,
                input,
            ),
            ReduceBenchKind::Fused => strategy_result_with_indices(
                client,
                problem.shape.clone(),
                problem.axis,
                *strategy,
                problem.operation,
                input,
            ),
        }
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

/// At most [`RAMP_MAX_ELEMS`] elements, all on the reduced axis so a plane or
/// cube routine has enough to reduce. Padding other axes with 2 keeps the total
/// a power of two at any rank, instead of the reduced axis underflowing to
/// zero once rank exceeds `RAMP_MAX_ELEMS`'s bit width. An axis of one in
/// `shape` stays one, so a whole tensor's one row is proved as one row.
fn proof_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    let mut proof = vec![1; shape.len()];
    let mut budget = RAMP_MAX_ELEMS;
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

impl cubek_test_utils::Correctness for ReduceCorrectness {
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
mod proof_shape_tests {
    use super::proof_shape;
    use crate::eval::cpu_reference::RAMP_MAX_ELEMS;

    /// A proof shape whose values a ramp permutes: a power of two at most the ramp's longest.
    fn fits_a_ramp(proof: &[usize]) -> bool {
        let elems: usize = proof.iter().product();
        elems.is_power_of_two() && elems <= RAMP_MAX_ELEMS
    }

    /// The fewest axes keep within a ramp.
    #[test]
    fn one_axis_stays_within_ramp_bounds() {
        assert!(fits_a_ramp(&proof_shape(&[3], 0)));
    }

    /// The most axes, reduced along the first, keep within a ramp.
    #[test]
    fn the_most_axes_along_the_first_stay_within_ramp_bounds() {
        const RANK: usize = 16;
        assert!(fits_a_ramp(&proof_shape(&[3; RANK], 0)));
    }

    /// The most axes, reduced along the last, keep within a ramp.
    #[test]
    fn the_most_axes_along_the_last_stay_within_ramp_bounds() {
        const RANK: usize = 16;
        assert!(fits_a_ramp(&proof_shape(&[3; RANK], RANK - 1)));
    }

    #[test]
    fn matches_ramp_max_elems_below_its_bit_width() {
        assert_eq!(proof_shape(&[3, 3, 3], 2), vec![2, 2, 512]);
    }

    #[test]
    fn keeps_a_whole_tensor_one_row() {
        assert_eq!(proof_shape(&[1, 16_777_216], 1), vec![1, 2048]);
    }
}
