use cubecl::{TestRuntime, prelude::*};
use cubek_matmul::{
    definition::MatmulElems,
    definition::{MatmulProblem, MatmulSetupError},
    launch::launch_ref,
    strategy::Strategy,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, LayoutSpec, StridedLayout, TestInput, TestOutcome, launch_and_capture_outcome,
};

use crate::matmul::assert_result;

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

pub(crate) fn run<F>(client: ComputeClient<TestRuntime>, problem: MatmulProblem, launch: F)
where
    F: FnOnce(
        &ComputeClient<TestRuntime>,
        InputBinding<TestRuntime>,
        InputBinding<TestRuntime>,
        TensorBinding<TestRuntime>,
        &mut MatmulElems,
    ) -> Result<(), MatmulSetupError>,
{
    let lhs_layout = problem.lhs_layout.into();
    let rhs_layout = problem.rhs_layout.into();
    run_outcome(client, problem, lhs_layout, rhs_layout, launch).enforce()
}

/// Like [`run`], but feeds the kernel the explicit strides already on `problem`
/// (via [`StridedLayout::Explicit`]) instead of deriving them from the layouts,
/// and returns the [`TestOutcome`] rather than enforcing it. Stride-0 broadcast
/// dims thus reach the kernel intact, and callers can interpret the outcome
/// themselves (e.g. compare a broadcast run against its contiguous baseline).
#[allow(unused)]
pub(crate) fn run_with_strides(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    strategy: Strategy,
) -> TestOutcome {
    let lhs_layout = StridedLayout::Explicit(problem.lhs_strides.to_vec()).into();
    let rhs_layout = StridedLayout::Explicit(problem.rhs_strides.to_vec()).into();
    run_outcome(
        client,
        problem,
        lhs_layout,
        rhs_layout,
        move |c, lhs, rhs, out, dtypes| launch_ref(&strategy, c, lhs, rhs, out, dtypes),
    )
}

/// Build the lhs/rhs inputs under the given layouts, launch via `launch`, and
/// return the [`TestOutcome`]. The built strides are written back onto `problem`
/// so the CPU reference sees the same memory layout the kernel did (a no-op when
/// the layouts already pin explicit strides).
fn run_outcome<F>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    lhs_layout: LayoutSpec,
    rhs_layout: LayoutSpec,
    launch: F,
) -> TestOutcome
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
        .layout(lhs_layout)
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(problem.global_dtypes.rhs)
        .layout(rhs_layout)
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();

    // Poisoned, not zeroed: the routine owns `out = A·B` whatever the buffer held
    // (burn launches with recycled pool memory).
    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(problem.global_dtypes.out)
        .layout(MatrixLayout::RowMajor)
        .uniform(4242, 10., 100.)
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
}
