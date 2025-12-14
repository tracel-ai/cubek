use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::components::MatrixLayout;
use cubek_matmul::components::global::args::TensorMapArgs;
use cubek_matmul::components::global::args::TensorMapInputs;

use cubek_matmul::components::global::args::ConcreteInputsFactory;
use cubek_matmul::components::{
    MatmulElems,
    global::args::{ConcreteOutputFactory, TensorArgs, TensorOutput},
};
use cubek_matmul::components::{MatmulProblem, MatmulSelection};
use cubek_matmul::components::{
    batch::{BatchConfig, BatchMatmulFamily},
    global::args::TensorInputs,
};
use cubek_matmul::kernels::layered::Algorithm;
use cubek_matmul::{
    MatmulInputHandleRef,
    components::{AvailableLineSizes, MatmulIdent},
};
use cubek_std::test_utils::HostData;
use cubek_std::test_utils::current_test_mode;
use cubek_std::test_utils::{Distribution, RandomInputSpec, SimpleInputSpec, TestInput};

use crate::suite::assert_result;

pub enum InputRepresentation {
    Normal,
    Tma,
}

#[allow(unused)]
/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_algorithm<A: Algorithm>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    selection: MatmulSelection,
    dtypes: MatmulElems,
    input_representation: InputRepresentation,
) {
    let lhs_shape = problem.shape(MatmulIdent::Lhs);
    let rhs_shape = problem.shape(MatmulIdent::Rhs);

    let (lhs, lhs_data) = TestInput::random(
        client.clone(),
        lhs_shape.clone(),
        *dtypes.lhs_global,
        1234,
        Distribution::Uniform(-1., 1.),
        problem.lhs_layout.into(),
    )
    .generate_with_f32_host_data()
    .unwrap();

    let (rhs, rhs_data) = TestInput::random(
        client.clone(),
        rhs_shape.clone(),
        *dtypes.rhs_global,
        5678,
        Distribution::Uniform(-1., 1.),
        problem.rhs_layout.into(),
    )
    .generate_with_f32_host_data()
    .unwrap();

    let out = TestInput::zeros(
        client.clone(),
        problem.shape(MatmulIdent::Out),
        *dtypes.acc_global,
    )
    .generate_without_host_data()
    .unwrap();

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let lhs_handle = MatmulInputHandleRef::Normal(lhs.as_ref(), *dtypes.lhs_global);
    let rhs_handle = MatmulInputHandleRef::Normal(rhs.as_ref(), *dtypes.rhs_global);
    let out_handle = out.as_ref();

    if launch_matmul_algorithm::<A>(
        &client,
        &problem,
        selection,
        &dtypes,
        input_representation,
        lhs_handle,
        rhs_handle,
        out_handle,
    ) {
        assert_result(&lhs_data, &rhs_data, &problem, &client, &out, dtypes);
    }
}

/// Returns whether execution succeeded
pub fn launch_matmul_algorithm<A: Algorithm>(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    selection: MatmulSelection,
    dtypes: &MatmulElems,
    input_representation: InputRepresentation,
    lhs: MatmulInputHandleRef<TestRuntime>,
    rhs: MatmulInputHandleRef<TestRuntime>,
    out: TensorHandleRef<TestRuntime>,
) -> bool {
    let line_sizes = AvailableLineSizes::from_type_sizes(
        &client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = match input_representation {
        InputRepresentation::Normal => line_sizes
            .filter_lhs_with_tensor(&lhs.data().strides, &lhs.data().shape, problem.lhs_layout)
            .filter_rhs_with_tensor(&rhs.data().strides, &rhs.data().shape, problem.rhs_layout)
            .filter_out_with_tensor(&out.strides, &out.shape)
            .pick_max()
            .unwrap(),
        InputRepresentation::Tma => line_sizes
            .filter_lhs(|ls| *ls == 1)
            .filter_rhs(|ls| *ls == 1)
            .pick_max()
            .unwrap(),
    };

    let config = match A::setup(&client, &problem, &selection, &line_sizes, &dtypes) {
        Ok(config) => config,
        Err(err) => {
            if current_test_mode().should_fail_on_test_compilation_fail() {
                panic!("Can't launch the test: {err}");
            }
            return false;
        }
    };

    let props = &client.properties().hardware;
    if !props.max_cube_dim.can_contain(config.cube_dim())
        || config.cube_dim().num_elems() > props.max_units_per_cube
    {
        println!("Skipping test, too many resources requested");
        return false;
    }

    let output = TensorOutput::create(
        &client,
        &out,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );

    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    match input_representation {
        InputRepresentation::Normal => {
            let inputs = TensorInputs::create(
                &client,
                &lhs,
                &rhs,
                &selection,
                &problem,
                &line_sizes,
                config,
                &dtypes,
            );

            unsafe {
                A::BatchMatmul::launch_unchecked::<TensorArgs, TestRuntime>(
                    &client,
                    config.cube_dim(),
                    cube_count_plan.resolve(),
                    inputs,
                    output,
                    cube_count_plan.as_args(),
                    config,
                    &dtypes,
                )
            }
        }
        InputRepresentation::Tma => {
            let inputs = TensorMapInputs::create(
                &client,
                &lhs,
                &rhs,
                &selection,
                &problem,
                &line_sizes,
                config,
                &dtypes,
            );

            unsafe {
                A::BatchMatmul::launch_unchecked::<TensorMapArgs, TestRuntime>(
                    &client,
                    config.cube_dim(),
                    cube_count_plan.resolve(),
                    inputs,
                    output,
                    cube_count_plan.as_args(),
                    config,
                    &dtypes,
                )
            }
        }
    }
    .is_ok()
}
