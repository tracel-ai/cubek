//! Shared test launcher.
//!
//! `test_algo` is a convenience wrapper that builds a 2D conv `Problem` from a
//! `ConvolutionSize` + tiling/swizzle/buffering, then runs the kernel against a
//! CPU reference via `cubek-test-utils`.
//!
//! TODO: route through the public `cubek_convolution::launch_ref` once the
//! blueprint surface is settled. For now we go straight to `launch_kernel`.

use cubecl::{
    TestRuntime,
    server::ServerError,
    zspace::{Shape, shape},
    {ir::AddressType, prelude::*},
};
use cubek_convolution::{
    components::{
        ConvolutionOperation, ConvolutionProblem, Dimensionality, global::args::RuntimeArgs,
    },
    forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
    kernels::algorithm::Algorithm,
};
use cubek_matmul::{
    components::{
        global::{InputLoadFlow, LoadFlows},
        stage::PartitionBuffering,
        tile_matmul::DispatchTileMatmul,
    },
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulGlobalElems, SwizzleModes, TilingBlueprint,
        TilingScheme,
    },
    launch::{InputArg, OutputArg},
    routines::{BlueprintStrategy, Routine},
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{DataKind, Distribution, StrideSpec, TestInput, TestOutcome};

use crate::suite::reference::assert_result;

/// 2D convolution input/output channel + spatial size, used by `test_algo` to
/// build a `ConvolutionProblem`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionSize {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub out_c: usize,
}

/// Build a 2D conv problem and run it. Used by both basic-tier helpers and the
/// macro-driven `full/` tier.
///
/// `dtypes` lets the caller decide the global/stage/register storage types
/// (e.g. f16-only vs f32 global with tf32 stage). The kernel uses it for
/// dispatch; the test harness uses `lhs_global` for sampling and the full
/// `MatmulElems` to compute a per-precision epsilon.
#[allow(clippy::too_many_arguments)]
pub fn test_algo<A: Algorithm<Routine: Routine<RuntimeArgs, Blueprint = TilingBlueprint>>>(
    dtypes: MatmulElems,
    tiling_scheme: TilingScheme,
    swizzle: SwizzleModes,
    partition_buffering: PartitionBuffering,
    convolution_size: ConvolutionSize,
) where
    A::Args: ConcreteArgs<A::Routine>,
{
    let client = TestRuntime::client(&Default::default());
    let plane_dim = client.properties().hardware.plane_size_max;

    // Fixed for now; mirrors the parameters baked into the previous test suite.
    let batches = 2;
    let kernel_size = vec![4, 3];
    let stride = vec![1, 1];
    let padding = vec![3, 1];
    let dilation = vec![3, 2];

    let out_h = calculate_conv_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        convolution_size.h,
    );
    let out_w = calculate_conv_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        convolution_size.w,
    );

    let lhs_layout = MatrixLayout::RowMajor;
    let rhs_layout = MatrixLayout::ColMajor;

    let m = batches * out_h * out_w;
    let k = kernel_size.iter().product::<u32>() as usize * convolution_size.c;
    let n = convolution_size.out_c;

    let lhs_strides = lhs_layout.to_strides(&vec![m, k]);
    let rhs_strides = rhs_layout.to_strides(&vec![k, n]);

    let global_dtypes = MatmulGlobalElems {
        lhs: dtypes.lhs_global,
        rhs: dtypes.rhs_global,
        out: dtypes.acc_global,
    };

    let mut problem = ConvolutionProblem {
        m,
        n,
        k,
        lhs_strides,
        rhs_strides,
        lhs_layout,
        rhs_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        batches,
        in_shape: shape![convolution_size.h, convolution_size.w],
        channels: convolution_size.c,
        out_channels: convolution_size.out_c,
        padded_channels: convolution_size.c,
        out_shape: shape![out_h, out_w],
        dimensionality: Dimensionality::Dim2,
        operation: ConvolutionOperation::Forward,
        global_dtypes,
        address_type: AddressType::U32,
    };

    let mut blueprint = TilingBlueprint::builder(
        DispatchTileMatmul::Cmma,
        tiling_scheme,
        plane_dim,
        &problem.as_matmul_problem(),
    )
    .shared_swizzle(swizzle)
    .partition_buffering(partition_buffering);

    if A::IS_SPECIALIZED {
        blueprint = blueprint.load_specialization_config(LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        });
    }

    let blueprint = blueprint.build();

    let lhs_shape: Shape = shape![
        problem.batches,
        problem.in_shape[0],
        problem.in_shape[1],
        problem.channels,
    ];
    let rhs_shape: Shape = shape![
        problem.n,
        problem.kernel_size[0] as usize,
        problem.kernel_size[1] as usize,
        problem.channels,
    ];
    let out_shape: Shape = shape![problem.batches, out_h, out_w, problem.n];

    let (lhs, lhs_data) = TestInput::new(
        client.clone(),
        lhs_shape,
        dtypes.lhs_global,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 1234,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::new(
        client.clone(),
        rhs_shape,
        dtypes.rhs_global,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 5678,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let out = TestInput::new(
        client.clone(),
        out_shape,
        dtypes.acc_global,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    // Update problem strides to the physical NHWC layout coming out of TestInput.
    problem.lhs_strides = lhs.strides().clone();
    problem.rhs_strides = rhs.strides().clone();

    let outcome = match launch_kernel::<A>(
        &client,
        &mut problem,
        blueprint,
        &dtypes,
        lhs.clone(),
        rhs.clone(),
        out.clone(),
    ) {
        Ok(()) => match get_server_error(&client) {
            Some(e) => e,
            None => TestOutcome::Validated(assert_result(
                &lhs_data, &rhs_data, &problem, &client, out, dtypes,
            )),
        },
        Err(e) => TestOutcome::CompileError(format!("{e}")),
    };

    outcome.enforce()
}

#[allow(clippy::too_many_arguments)]
fn launch_kernel<A: Algorithm<Routine: Routine<RuntimeArgs, Blueprint = TilingBlueprint>>>(
    client: &ComputeClient<TestRuntime>,
    problem: &mut ConvolutionProblem,
    blueprint: <A::Routine as Routine<RuntimeArgs>>::Blueprint,
    dtypes: &MatmulElems,
    lhs: cubecl::std::tensor::TensorHandle<TestRuntime>,
    rhs: cubecl::std::tensor::TensorHandle<TestRuntime>,
    out: cubecl::std::tensor::TensorHandle<TestRuntime>,
) -> Result<(), cubek_matmul::definition::MatmulSetupError>
where
    A::Args: ConcreteArgs<A::Routine>,
{
    let vector_sizes = AvailableVectorSizes {
        lhs: vec![1],
        rhs: vec![1],
        out: client
            .io_optimized_vector_sizes(dtypes.acc_global.size())
            .collect(),
    }
    .filter_lhs_with_tensor(lhs.strides(), lhs.shape(), problem.lhs_layout)
    .filter_rhs_with_tensor(rhs.strides(), rhs.shape(), problem.rhs_layout)
    .filter_out_with_tensor(out.strides(), out.shape())
    .pick_max()
    .unwrap();

    let device_settings = A::Routine::device_settings(client, vector_sizes);
    let expand_info = A::Routine::expand_blueprint(
        &problem.as_matmul_problem(),
        &device_settings,
        &BlueprintStrategy::Forced(blueprint),
    )?;
    let problem_adjusted =
        A::Args::adjust_problem(client, problem.clone(), &expand_info.blueprint, dtypes);

    let launch_info = A::Routine::prepare(
        &problem_adjusted.as_matmul_problem(),
        &device_settings,
        expand_info,
    )?;

    let op = ConvolutionOperation::Forward;

    let lhs_handle = A::correct_layout(client, lhs.binding(), dtypes.lhs_global, op).unwrap();
    let rhs_handle = A::correct_layout(client, rhs.binding(), dtypes.rhs_global, op).unwrap();

    let lhs_handle = InputBinding::new(lhs_handle, dtypes.lhs_global);
    let rhs_handle = InputBinding::new(rhs_handle, dtypes.rhs_global);

    let (inputs, runtime_args) = <InputArg<A::Args> as ConcreteInputsFactory<A::Routine>>::create(
        lhs_handle,
        rhs_handle,
        None,
        &launch_info.blueprint,
        &problem_adjusted,
        dtypes,
    );
    let output = <OutputArg<A::Args> as ConcreteOutputFactory<A::Routine>>::create(
        out.binding(),
        &launch_info.blueprint,
        &problem_adjusted,
        dtypes,
    );

    cubek_matmul::launch::launch_kernel::<A::Args, TestRuntime, A::Routine>(
        client,
        inputs,
        output,
        runtime_args,
        launch_info,
    )
}

fn get_server_error(client: &ComputeClient<TestRuntime>) -> Option<TestOutcome> {
    match client.flush() {
        Ok(_) => None,
        Err(ServerError::ServerUnhealthy { errors, .. }) => {
            #[allow(clippy::never_loop)]
            for error in errors.iter() {
                match error {
                    cubecl::server::ServerError::Launch(LaunchError::TooManyResources(_))
                    | cubecl::server::ServerError::Launch(LaunchError::CompilationError(_)) => {
                        return Some(TestOutcome::CompileError(format!("{errors:?}")));
                    }
                    _ => panic!("Unexpected error: {errors:?}"),
                }
            }

            None
        }
        Err(err) => panic!("Unexpected error: {err:?}"),
    }
}

/// Calculate the expected output size when doing a convolution operation.
pub fn calculate_conv_output_size(
    kernel_size: u32,
    stride: u32,
    padding: i32,
    dilation: u32,
    size_in: usize,
) -> usize {
    (size_in + 2 * padding as usize - dilation as usize * (kernel_size as usize - 1) - 1)
        / stride as usize
        + 1
}
