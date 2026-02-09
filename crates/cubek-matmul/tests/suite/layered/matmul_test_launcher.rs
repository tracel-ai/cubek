use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::definition::AvailableLineSizes;
use cubek_matmul::definition::MatmulIdent;
use cubek_matmul::definition::MatrixLayout;
use cubek_matmul::launch::ConcreteOutputFactory;
use cubek_matmul::launch::ConcreteOutputFactory as _;

use cubek_matmul::components::batch::{BatchConfig, BatchMatmulFamily};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::definition::{MatmulProblem, TilingBlueprint};
use cubek_matmul::launch::ConcreteInputsFactory;
use cubek_matmul::launch::MatmulInputHandleRef;
use cubek_matmul::launch::TensorArgs;
use cubek_matmul::launch::TensorInputs;
use cubek_matmul::launch::TensorMapArgs;
use cubek_matmul::launch::TensorMapInputs;
use cubek_matmul::launch::TensorOutput;
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::Routine;
use cubek_test_utils::DataKind;
use cubek_test_utils::ExecutionOutcome;
use cubek_test_utils::HostData;
use cubek_test_utils::TestOutcome;
use cubek_test_utils::current_test_mode;
use cubek_test_utils::{BaseInputSpec, Distribution, RandomInputSpec, TestInput};

use crate::suite::assert_result;
use crate::suite::layout_to_stride_spec;

pub enum InputRepresentation {
    Normal,
    Tma,
}

#[allow(unused)]
/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_algorithm<A: Routine<(), Blueprint = TilingBlueprint>>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    blueprint: A::Blueprint,
    input_representation: InputRepresentation,
) {
    let (lhs, lhs_data) = TestInput::new(
        client.clone(),
        problem.lhs_shape.clone(),
        problem.global_dtypes.lhs,
        layout_to_stride_spec(problem.lhs_layout),
        DataKind::Random {
            seed: 1234,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::new(
        client.clone(),
        problem.rhs_shape.clone(),
        problem.global_dtypes.rhs,
        layout_to_stride_spec(problem.rhs_layout),
        DataKind::Random {
            seed: 1234,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let out = TestInput::new(
        client.clone(),
        problem.out_shape.clone(),
        problem.global_dtypes.out,
        layout_to_stride_spec(MatrixLayout::RowMajor),
        DataKind::Zeros,
    )
    .generate_without_host_data();

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let lhs_handle = MatmulInputHandleRef::Normal(lhs.as_ref(), problem.global_dtypes.lhs);
    let rhs_handle = MatmulInputHandleRef::Normal(rhs.as_ref(), problem.global_dtypes.rhs);
    let out_handle = out.as_ref();

    let all_elems = MatmulElems::from_globals(&problem.global_dtypes.clone());

    match launch_matmul_algorithm::<A>(
        &client,
        &problem,
        blueprint,
        &all_elems,
        input_representation,
        lhs_handle,
        rhs_handle,
        out_handle,
    ) {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_data, &rhs_data, &problem, &client, &out, all_elems)
                .as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}

/// Returns whether execution succeeded
#[allow(clippy::too_many_arguments)]
pub fn launch_matmul_algorithm<A: Routine<(), Blueprint = TilingBlueprint>>(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    blueprint: A::Blueprint,
    dtypes: &MatmulElems,
    input_representation: InputRepresentation,
    lhs: MatmulInputHandleRef<TestRuntime>,
    rhs: MatmulInputHandleRef<TestRuntime>,
    out: TensorHandleRef<TestRuntime>,
) -> ExecutionOutcome {
    let line_sizes = AvailableLineSizes::from_type_sizes(
        client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    );
    let line_sizes = match input_representation {
        InputRepresentation::Normal => line_sizes
            .filter_lhs_with_tensor(lhs.data().strides, lhs.data().shape, problem.lhs_layout)
            .filter_rhs_with_tensor(rhs.data().strides, rhs.data().shape, problem.rhs_layout)
            .filter_out_with_tensor(out.strides, out.shape)
            .pick_max()
            .unwrap(),
        InputRepresentation::Tma => line_sizes
            .filter_lhs(|ls| *ls == 1)
            .filter_rhs(|ls| *ls == 1)
            .pick_max()
            .unwrap(),
    };

    let device_settings = A::device_settings(client, line_sizes);

    let expand_info = match A::expand_blueprint(
        problem,
        &device_settings,
        &BlueprintStrategy::Forced(blueprint),
    ) {
        Ok(launch_info) => launch_info,
        Err(err) => {
            return ExecutionOutcome::CompileError(format!("Can't launch the test: {err}"));
        }
    };

    let launch_info = match A::prepare(
        problem,
        &A::device_settings(client, line_sizes),
        expand_info,
    ) {
        Ok(launch_info) => launch_info,
        Err(err) => {
            return ExecutionOutcome::CompileError(format!("Can't launch the test: {err}"));
        }
    };

    let cube_dim = launch_info.cube_dim;
    let cube_count_plan = launch_info.cube_count_plan;
    let blueprint = launch_info.blueprint;
    let dtypes = &launch_info.dtypes;

    let output = <TensorOutput<_> as ConcreteOutputFactory<A>>::create(
        client,
        &out,
        &blueprint,
        problem,
        &line_sizes,
        dtypes,
    );

    match input_representation {
        InputRepresentation::Normal => {
            let inputs = <TensorInputs<_, _, _> as ConcreteInputsFactory<A>>::create(
                client,
                &lhs,
                &rhs,
                &blueprint,
                problem,
                &line_sizes,
                dtypes,
            );

            unsafe {
                A::BatchMatmul::launch_unchecked::<TensorArgs, TestRuntime>(
                    client,
                    cube_dim,
                    cube_count_plan.resolve(),
                    inputs,
                    output,
                    (),
                    cube_count_plan.as_args(),
                    blueprint,
                    dtypes,
                )
            }
        }
        InputRepresentation::Tma => {
            let inputs = <TensorMapInputs<_, _, _> as ConcreteInputsFactory<A>>::create(
                client,
                &lhs,
                &rhs,
                &blueprint,
                problem,
                &line_sizes,
                dtypes,
            );

            unsafe {
                A::BatchMatmul::launch_unchecked::<TensorMapArgs, TestRuntime>(
                    client,
                    cube_dim,
                    cube_count_plan.resolve(),
                    inputs,
                    output,
                    (),
                    cube_count_plan.as_args(),
                    blueprint,
                    dtypes,
                )
            }
        }
    }
    .into()
}
