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
use cubek_matmul::launch::Strategy;
use cubek_matmul::launch::TensorArgs;
use cubek_matmul::launch::TensorInputs;
use cubek_matmul::launch::TensorMapArgs;
use cubek_matmul::launch::TensorMapInputs;
use cubek_matmul::launch::TensorOutput;
use cubek_matmul::launch::launch_ref;
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

#[allow(unused)]
/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_strategy(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    strategy: Strategy,
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

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    launch_ref(
        &strategy,
        &client,
        lhs_handle,
        rhs_handle,
        out_handle,
        &mut dtypes,
    );
}
