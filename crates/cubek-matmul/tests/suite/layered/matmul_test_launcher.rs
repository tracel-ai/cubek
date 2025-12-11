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
use cubek_std::test_utils::compute_strides;
use cubek_std::test_utils::random_tensor;

use crate::suite::assert_result;

pub enum InputRepresentation {
    Normal,
    Tma,
}

// TODO should be always used, remove some feature flags
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

    let (lhs, lhs_data) = random_tensor(
        &client,
        *dtypes.lhs_global,
        1234,
        compute_strides(
            &lhs_shape,
            matches!(problem.lhs_layout, MatrixLayout::ColMajor),
        ),
        lhs_shape,
    );
    let (rhs, rhs_data) = random_tensor(
        &client,
        *dtypes.rhs_global,
        5678,
        compute_strides(
            &rhs_shape,
            matches!(problem.rhs_layout, MatrixLayout::ColMajor),
        ),
        rhs_shape,
    );
    let out = TensorHandle::zeros(&client, problem.shape(MatmulIdent::Out), *dtypes.acc_global);

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

    let env = std::env::var("CUBEK_TEST_MODE");

    let panic_on_launch_err = match env {
        Ok(val) => match val.as_str() {
            "panic" => true,
            "skip" => false,
            _ => false,
        },
        Err(_) => false,
    };

    let config = match A::setup(&client, &problem, &selection, &line_sizes, &dtypes) {
        Ok(config) => config,
        Err(err) => {
            let msg = format!("Can't launch the test: {err}");
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
                return false;
            }
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
