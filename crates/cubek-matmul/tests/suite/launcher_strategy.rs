use cubecl::{TestRuntime, prelude::*};
use cubek_matmul::{
    definition::MatmulElems,
    definition::{MatmulProblem, MatmulSetupError},
    launch::Strategy,
    launch::launch_ref,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};

use crate::{suite::assert_result, suite::layout_to_stride_spec};

/// Test the correctness of a public [`Strategy`] against the CPU reference.
#[allow(unused)]
pub fn test_matmul_strategy(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    strategy: Strategy,
) {
    run(client, problem, move |client, lhs, rhs, out, dtypes| {
        launch_ref(&strategy, client, lhs, rhs, out, dtypes)
    });
}

pub(crate) fn run<F>(client: ComputeClient<TestRuntime>, mut problem: MatmulProblem, launch: F)
where
    F: FnOnce(
        &ComputeClient<TestRuntime>,
        InputBinding<TestRuntime>,
        InputBinding<TestRuntime>,
        TensorBinding<TestRuntime>,
        &mut MatmulElems,
    ) -> Result<(), MatmulSetupError>,
{
    let (lhs, lhs_data) = TestInput::builder(client.clone(), problem.lhs_shape.clone())
        .dtype(problem.global_dtypes.lhs)
        .stride(layout_to_stride_spec(problem.lhs_layout))
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(problem.global_dtypes.rhs)
        .stride(layout_to_stride_spec(problem.rhs_layout))
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();

    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(problem.global_dtypes.out)
        .stride(layout_to_stride_spec(MatrixLayout::RowMajor))
        .zeros()
        .generate_without_host_data();

    problem.lhs_strides = lhs.strides().clone();
    problem.rhs_strides = rhs.strides().clone();

    let lhs_handle = InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    let out_handle = out.clone().binding();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    let outcome = launch_and_capture_outcome(&client, |c| {
        launch(c, lhs_handle, rhs_handle, out_handle, &mut dtypes).into()
    });

    match outcome {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_data, &rhs_data, &problem, &client, out, dtypes).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce()
}
