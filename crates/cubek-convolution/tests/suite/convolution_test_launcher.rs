use crate::suite::test_utils::{Sample, TensorRawParts};
use cubecl::{CubeElement, server::Allocation};
use cubecl::{TestRuntime, prelude::*};
use cubek_convolution::{
    algorithm::Algorithm,
    components::{ConvolutionProblem, global::args::RuntimeArgs},
};
use cubek_convolution::{
    components::{ConvGemmConfig, ConvSetupError, ConvolutionOperation},
    forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
};
use cubek_matmul::{
    definition::{AvailableLineSizes, MatmulSetupError},
    launch::MatmulInputHandleRef,
};
use cubek_matmul::{
    definition::{MatmulElems, MatmulIdent, TilingBlueprint},
    routines::Routine,
};
use cubek_matmul::{
    launch::{InputArg, OutputArg},
    routines::BlueprintStrategy,
};

use super::test_utils::TestPrecision;

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_convolution_algorithm<A: Algorithm, P: TestPrecision>(
    client: ComputeClient<TestRuntime>,
    mut problem: ConvolutionProblem,
    blueprint: <A::Routine as Routine<RuntimeArgs>>::Blueprint,
) where
    A::Args: ConcreteArgs<A::Routine>,
{
    let env = std::env::var("CUBEK_TEST_MODE");

    let panic_on_launch_err = match env {
        Ok(val) => match val.as_str() {
            "panic" => true,
            "skip" => false,
            _ => false,
        },
        Err(_) => false,
    };

    let result = test_convolution_algorithm_inner::<A, P>(client, problem, blueprint);

    match result {
        Ok(_) => {}
        Err(err) => {
            let msg = format!("Can't launch the test: {err}");
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
            }
        }
    }
}

fn test_convolution_algorithm_inner<A: Algorithm, P: TestPrecision>(
    client: ComputeClient<TestRuntime>,
    mut problem: ConvolutionProblem,
    blueprint: <A::Routine as Routine<RuntimeArgs>>::Blueprint,
) -> Result<(), MatmulSetupError>
where
    A::Args: ConcreteArgs<A::Routine>,
{
    let lhs = tensor_raw_parts::<P, TestRuntime>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<P, TestRuntime>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<P, TestRuntime>(&client, &problem, MatmulIdent::Out);

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let line_sizes = AvailableLineSizes {
        lhs: vec![1],
        rhs: vec![1],
        out: client
            .io_optimized_line_sizes_unchecked(size_of::<P::EG>())
            .collect(),
    }
    .filter_lhs_with_tensor(&lhs.strides, &lhs.shape, problem.lhs_layout)
    .filter_rhs_with_tensor(&rhs.strides, &rhs.shape, problem.rhs_layout)
    .filter_out_with_tensor(&out.strides, &out.shape)
    .pick_max()
    .unwrap();

    let dtypes = MatmulElems::new_deprecated::<((P::EG, P::ES), (P::EG, P::ES), (P::EG, f32))>();

    let device_settings = A::Routine::device_settings(&client, line_sizes);
    let expand_info = A::Routine::expand_blueprint(
        &problem.as_matmul_problem(),
        &device_settings,
        &BlueprintStrategy::Forced(blueprint),
    )?;
    let problem = A::Args::adjust_problem(&client, problem, &expand_info.blueprint, &dtypes);

    let launch_info =
        A::Routine::prepare(&problem.as_matmul_problem(), &device_settings, expand_info)?;

    let elem_size = size_of::<P::EG>();
    let lhs_handle = unsafe {
        TensorHandleRef::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, elem_size)
    };
    let rhs_handle = unsafe {
        TensorHandleRef::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, elem_size)
    };
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(&out.handle, &out.strides, &out.shape, elem_size)
    };

    let op = ConvolutionOperation::Forward;

    let lhs_handle =
        A::into_tensor_handle(&client, &lhs_handle, P::EG::as_type_native_unchecked(), op).unwrap();
    let rhs_handle =
        A::into_tensor_handle(&client, &rhs_handle, P::EG::as_type_native_unchecked(), op).unwrap();

    let lhs_handle =
        MatmulInputHandleRef::new(lhs_handle.as_ref(), P::EG::as_type_native_unchecked());
    let rhs_handle =
        MatmulInputHandleRef::new(rhs_handle.as_ref(), P::EG::as_type_native_unchecked());

    let (inputs, runtime_args) = <InputArg<A::Args> as ConcreteInputsFactory<A::Routine>>::create(
        &client,
        &lhs_handle,
        &rhs_handle,
        None,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        &dtypes,
    );
    let output = <OutputArg<A::Args> as ConcreteOutputFactory<A::Routine>>::create(
        &client,
        &out_handle,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        &dtypes,
    );

    let dtypes = MatmulElems::new_deprecated::<((P::EG, P::ES), (P::EG, P::ES), (P::EG, P::EA))>();

    cubek_matmul::launch::launch_kernel::<A::Args, TestRuntime, A::Routine>(
        &client,
        inputs,
        output,
        runtime_args,
        launch_info,
    )?;

    P::assert_result(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
        &out.shape,
        &out.strides,
    );

    Ok(())
}

fn tensor_raw_parts<P: TestPrecision, R: Runtime>(
    client: &ComputeClient<R>,
    problem: &ConvolutionProblem,
    ident: MatmulIdent,
) -> TensorRawParts<P::EG> {
    match ident {
        MatmulIdent::Lhs => {
            let shape = shape(problem, ident);

            let handle = P::EG::sample(client, &shape, 1234);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            TensorRawParts {
                handle: handle.handle,
                scale: None,
                shape,
                strides: handle.strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Rhs => {
            let shape = shape(problem, ident);

            let handle = P::EG::sample(client, &shape, 1234);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
            let data = P::EG::from_bytes(&data);
            let original_data = data.to_owned();

            TensorRawParts {
                handle: handle.handle,
                scale: None,
                shape,
                strides: handle.strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Out => {
            let zero = P::EG::from_int(0);

            let data = vec![zero; tensor_size(problem, MatmulIdent::Out)];

            let shape = shape(problem, MatmulIdent::Out);
            let Allocation { handle, strides } =
                client.create_tensor_from_slice(P::EG::as_bytes(&data), &shape, size_of::<P::EG>());

            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: None,
            }
        }
    }
}

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &ConvolutionProblem, ident: MatmulIdent) -> usize {
    match ident {
        MatmulIdent::Lhs => problem.m * problem.k,
        MatmulIdent::Rhs => problem.k * problem.n,
        MatmulIdent::Out => problem.m * problem.n,
    }
}

/// Returns the shape of the identified tensor, inferred by the problem definition
pub(crate) fn shape(problem: &ConvolutionProblem, ident: MatmulIdent) -> Vec<usize> {
    match ident {
        MatmulIdent::Lhs => vec![
            problem.batches,
            problem.in_shape[0],
            problem.in_shape[1],
            problem.channels,
        ],
        MatmulIdent::Rhs => vec![
            problem.n,
            problem.kernel_size[0] as usize,
            problem.kernel_size[1] as usize,
            problem.channels,
        ],
        MatmulIdent::Out => vec![
            problem.batches,
            problem.out_shape[0],
            problem.out_shape[1],
            problem.n,
        ],
    }
}
