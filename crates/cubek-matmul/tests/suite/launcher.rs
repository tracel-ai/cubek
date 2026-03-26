use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubecl::server::ServerError;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::definition::AvailableVectorSizes;
use cubek_matmul::definition::MatmulIdent;
use cubek_matmul::definition::MatmulVectorSizes;
use cubek_matmul::definition::cube_mapping_launch;
use cubek_matmul::launch::ConcreteOutputFactory;
use cubek_matmul::launch::ConcreteOutputFactory as _;

use cubek_matmul::components::batch::{BatchConfig, BatchMatmulFamily};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::definition::{MatmulProblem, TilingBlueprint};
use cubek_matmul::launch::ConcreteInputsFactory;
use cubek_matmul::launch::TensorArgs;
use cubek_matmul::launch::TensorInputs;
use cubek_matmul::launch::TensorMapArgs;
use cubek_matmul::launch::TensorMapInputs;
use cubek_matmul::launch::TensorOutput;
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::Routine;
use cubek_std::InputBinding;
use cubek_std::MatrixLayout;
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
pub fn test_matmul_algorithm<A: Routine<()>>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    blueprint_strategy: BlueprintStrategy<(), A>,
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
            seed: 5678,
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

    problem.lhs_strides = lhs.strides().clone();
    problem.rhs_strides = rhs.strides().clone();

    let lhs_handle = InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    let out_handle = out.clone().binding();

    let all_elems = MatmulElems::from_globals(&problem.global_dtypes.clone());

    match launch_matmul_algorithm::<A>(
        &client,
        &problem,
        blueprint_strategy,
        &all_elems,
        input_representation,
        lhs_handle,
        rhs_handle,
        out_handle,
    ) {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_data, &rhs_data, &problem, &client, out, all_elems).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}

/// Returns whether execution succeeded
#[allow(clippy::too_many_arguments)]
pub fn launch_matmul_algorithm<A: Routine<()>>(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    blueprint_strategy: BlueprintStrategy<(), A>,
    dtypes: &MatmulElems,
    input_representation: InputRepresentation,
    lhs: InputBinding<TestRuntime>,
    rhs: InputBinding<TestRuntime>,
    out: TensorBinding<TestRuntime>,
) -> ExecutionOutcome {
    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    );
    let vector_sizes = match input_representation {
        InputRepresentation::Normal => vector_sizes
            .filter_lhs_with_tensor(&lhs.data().strides, &lhs.data().shape, problem.lhs_layout)
            .filter_rhs_with_tensor(&rhs.data().strides, &rhs.data().shape, problem.rhs_layout)
            .filter_out_with_tensor(&out.strides, &out.shape)
            .pick_max()
            .unwrap(),
        InputRepresentation::Tma => vector_sizes
            .filter_lhs(|ls| *ls == 1)
            .filter_rhs(|ls| *ls == 1)
            .pick_max()
            .unwrap(),
    };

    let device_settings = A::device_settings(client, vector_sizes);

    let expand_info = match A::expand_blueprint(problem, &device_settings, &blueprint_strategy) {
        Ok(launch_info) => launch_info,
        Err(err) => {
            return ExecutionOutcome::CompileError(format!("Can't launch the test: {err}"));
        }
    };

    let launch_info = match A::prepare(
        problem,
        &A::device_settings(client, vector_sizes),
        expand_info,
    ) {
        Ok(launch_info) => launch_info,
        Err(err) => {
            return ExecutionOutcome::CompileError(format!("Can't launch the test: {err}"));
        }
    };

    let client = client.clone();
    let problem = problem.clone();
    let client_cloned = client.clone();
    let cube_dim = launch_info.cube_dim;
    let cube_count_plan = launch_info.cube_count_plan;
    let blueprint = launch_info.blueprint;
    let dtypes = &launch_info.dtypes.clone();

    let output = <TensorOutput<_> as ConcreteOutputFactory<A>>::create(
        out,
        &blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );

    let result = match input_representation {
        InputRepresentation::Normal => {
            let inputs = <TensorInputs<_, _, _> as ConcreteInputsFactory<A>>::create(
                lhs,
                rhs,
                &blueprint,
                &problem,
                &vector_sizes,
                dtypes,
            );

            unsafe {
                A::BatchMatmul::launch_unchecked::<TensorArgs, TestRuntime>(
                    &client,
                    cube_dim,
                    cube_count_plan.resolve(),
                    AddressType::U32,
                    inputs,
                    output,
                    (),
                    cube_mapping_launch(&cube_count_plan),
                    blueprint,
                    dtypes,
                    &vector_sizes,
                )
            }
        }
        InputRepresentation::Tma => {
            let inputs = <TensorMapInputs<_, _, _> as ConcreteInputsFactory<A>>::create(
                lhs,
                rhs,
                &blueprint,
                &problem,
                &vector_sizes,
                dtypes,
            );

            unsafe {
                A::BatchMatmul::launch_unchecked::<TensorMapArgs, TestRuntime>(
                    &client,
                    cube_dim,
                    cube_count_plan.resolve(),
                    AddressType::U32,
                    inputs,
                    output,
                    (),
                    cube_mapping_launch(&cube_count_plan),
                    blueprint,
                    dtypes,
                    &vector_sizes,
                )
            }
        }
    }
    .into();

    match client.flush() {
        Ok(_) => {}
        Err(ServerError::ServerUnhealthy { errors, .. }) =>
        {
            #[allow(clippy::never_loop)]
            for error in errors.iter() {
                match error {
                    cubecl::server::ServerError::Launch(LaunchError::TooManyResources(_)) => {
                        return ExecutionOutcome::CompileError(format!("{errors:?}"));
                    }
                    _ => panic!("{errors:?}"),
                }
            }
        }
        Err(err) => panic!("{err:?}"),
    }

    result
}
