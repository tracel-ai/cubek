//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubecl::client::Client;
use cubek_matmul::{
    definition::{MatmulElems, MatmulProblem},
    launch::launch_ref,
    multi_level::Strategy as MultiLevel,
};
use cubek_std::InputBinding;
use cubek_test_utils::{
    ExecutionOutcome, TestInput, TestOutcome, ValidationResult, launch_and_capture_outcome,
};

use crate::harness::{
    assert_result, client, f16_elems, f32_elems, rect, square, test_matmul_strategy,
};

/// Binary fractions survive TF32 conversion, so these layout tests keep the f32 tolerance.
fn run_exact_f32(client: Client, problem: MatmulProblem) -> TestOutcome {
    let (lhs, lhs_data) = TestInput::builder(client.clone(), problem.lhs_shape.clone())
        .dtype(problem.global_dtypes.lhs)
        .custom(
            (0..problem.lhs_shape.num_elements())
                .map(|index| ((index % 17) as f32 - 8.) / 8.)
                .collect(),
        )
        .generate_with_f32_host_data();
    let (rhs, rhs_data) = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(problem.global_dtypes.rhs)
        .custom(
            (0..problem.rhs_shape.num_elements())
                .map(|index| (((index * 3 + 1) % 17) as f32 - 8.) / 8.)
                .collect(),
        )
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(problem.global_dtypes.out)
        .uniform(4242, 10., 100.)
        .generate_without_host_data();
    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
    let strategy = MultiLevel::SimpleCyclicCmma(Default::default()).into();
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
        launch_ref(
            &strategy,
            c,
            InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs),
            InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs),
            out.clone().binding(),
            &mut dtypes,
        )
        .into()
    });
    match outcome {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_data, &rhs_data, &problem, &client, out, dtypes).as_test_outcome()
        }
        ExecutionOutcome::CompileError(error) => TestOutcome::CompileError(error),
    }
}

#[test]
fn simple_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicCmma(Default::default()).into(),
    );
}

/// A small f32 problem runs rather than declining: a routine that runs at 37 rows also runs at 32,
/// since autotune buckets the two together. Asserted as executed on a device with cmma, where the
/// default test policy would accept the decline; reported as skipped on one without.
#[test]
fn simple_cyclic_cmma_runs_a_small_f32_problem() {
    let client = client();
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the device offers no cmma instruction".to_string(),
        ))
        .enforce();
        return;
    }
    let outcome = run_exact_f32(client, rect(32, 32, 64, f32_elems()));
    assert!(
        matches!(outcome, TestOutcome::Validated(ValidationResult::Pass)),
        "expected the matmul to execute and validate, got {outcome:?}"
    );
}

#[test]
fn simple_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleCyclicMma(Default::default()).into(),
    );
}

#[test]
fn simple_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleStridedCmma(Default::default()).into(),
    );
}

#[test]
fn simple_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleStridedMma(Default::default()).into(),
    );
}

#[test]
fn simple_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTilewiseCmma(Default::default()).into(),
    );
}

#[test]
fn simple_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTilewiseMma(Default::default()).into(),
    );
}

#[test]
fn simple_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncStridedCmma(Default::default()).into(),
    );
}

#[test]
fn simple_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncStridedMma(Default::default()).into(),
    );
}

#[test]
fn simple_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn simple_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleAsyncCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn double_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_tilewise_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTilewiseCmma(Default::default()).into(),
    );
}

#[test]
fn double_tilewise_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleTilewiseMma(Default::default()).into(),
    );
}

#[test]
fn double_hybrid_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleHybridCmma(Default::default()).into(),
    );
}

#[test]
fn double_hybrid_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleHybridMma(Default::default()).into(),
    );
}

#[test]
fn double_async_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn double_async_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncCyclicMma(Default::default()).into(),
    );
}

#[test]
fn double_async_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncStridedCmma(Default::default()).into(),
    );
}

#[test]
fn double_async_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::DoubleAsyncStridedMma(Default::default()).into(),
    );
}

#[test]
fn specialized_cyclic_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedCyclicCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_cyclic_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedCyclicMma(Default::default()).into(),
    );
}

#[test]
fn specialized_strided_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedStridedCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_strided_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedStridedMma(Default::default()).into(),
    );
}

#[test]
fn ordered_double_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::OrderedDoubleCmma(Default::default()).into(),
    );
}

#[test]
fn ordered_double_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::OrderedDoubleMma(Default::default()).into(),
    );
}

#[test]
fn simple_cyclic_cmma_small_k_and_partial_line_width_rows_stay_correct() {
    use crate::harness::{f32_elems, rect};
    for _ in 0..20 {
        for (m, n, k) in [(382, 10, 1), (382, 10, 3), (4096, 14, 5)] {
            run_exact_f32(client(), rect(m, n, k, f32_elems())).enforce();
        }
    }
}
