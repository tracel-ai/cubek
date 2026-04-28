//! Shared test launcher.
//!
//! `test_algo` is a convenience wrapper that builds a 2D conv `Problem` from a
//! `ConvolutionSize` + tiling/swizzle/buffering, then routes through the public
//! `cubek_convolution::launch_ref` (`Strategy::Forced`) and validates the
//! output against a CPU reference via `cubek-test-utils`.

use cubecl::{
    TestRuntime,
    server::ServerError,
    zspace::{Shape, shape},
    {ir::AddressType, prelude::*},
};
use cubek_convolution::{
    ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    definition::{ConvBlueprint, ForwardBlueprint},
};
use cubek_matmul::{
    components::{
        global::{InputLoadFlow, LoadFlows},
        stage::PartitionBuffering,
        tile_matmul::TileMatmulKind,
    },
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulGlobalElems, TilingBlueprint, TilingScheme,
    },
    launch::{InputArg, OutputArg},
    routines::{BlueprintStrategy, Routine},
};
use cubek_std::{InputBinding, MatrixLayout, SwizzleModes};
use cubek_test_utils::{TestInput, TestOutcome};

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

/// Build a 2D forward conv problem and run it via the public `launch_ref`.
/// Used by both basic-tier helpers and the macro-driven `full/` tier.
///
/// `algorithm` selects the kernel reading-strategy variant. `dtypes` lets the
/// caller decide global/stage/register storage types. The kernel uses dtypes
/// for dispatch; the test harness uses `lhs_global` for sampling and the full
/// `MatmulElems` to compute a per-precision epsilon.
#[allow(clippy::too_many_arguments)]
pub fn test_algo(
    algorithm: ConvAlgorithm,
    dtypes: MatmulElems,
    tiling_scheme: TilingScheme,
    swizzle: SwizzleModes,
    partition_buffering: PartitionBuffering,
    convolution_size: ConvolutionSize,
) {
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

    // Build a synthetic problem just to compute the matmul layout — the actual
    // problem will be reconstructed inside `launch_ref` from the bindings.
    let problem = ConvolutionProblem {
        m,
        n,
        k,
        lhs_strides,
        rhs_strides,
        lhs_layout,
        rhs_layout,
        kernel_size: kernel_size.clone(),
        stride: stride.clone(),
        padding: padding.clone(),
        dilation: dilation.clone(),
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

    let matmul_blueprint = TilingBlueprint::builder(
        TileMatmulKind::Cmma,
        tiling_scheme,
        plane_dim,
        &problem.as_matmul_problem(),
    )
    .shared_swizzle(swizzle)
    .partition_buffering(partition_buffering)
    .build();

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

    let (lhs, lhs_data) = TestInput::builder(client.clone(), lhs_shape)
        .dtype(dtypes.lhs_global)
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::builder(client.clone(), rhs_shape)
        .dtype(dtypes.rhs_global)
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();

    let out = TestInput::builder(client.clone(), out_shape)
        .dtype(dtypes.acc_global)
        .zeros()
        .generate_without_host_data();

    let blueprint = ConvBlueprint::Forward(ForwardBlueprint {
        matmul: matmul_blueprint,
        dimensionality: Dimensionality::Dim2,
        has_bias: false,
    });
    let strategy = Strategy::Forced {
        algorithm,
        blueprint,
    };

    let inputs = ConvolutionInputs::Forward {
        input: InputBinding::new(lhs.clone().binding(), dtypes.lhs_global),
        weight: InputBinding::new(rhs.clone().binding(), dtypes.rhs_global),
        bias: None,
        out: out.clone().binding(),
    };

    let args = ConvolutionArgs::<2> {
        stride: [stride[0] as usize, stride[1] as usize],
        padding: [padding[0] as usize, padding[1] as usize],
        dilation: [dilation[0] as usize, dilation[1] as usize],
    };

    // Re-build problem with physical strides from the test inputs so the CPU
    // reference uses the same layout as the kernel sees.
    let mut problem_for_check = problem.clone();
    problem_for_check.lhs_strides = lhs.strides().clone();
    problem_for_check.rhs_strides = rhs.strides().clone();

    let outcome =
        match cubek_convolution::launch_ref(&strategy, &client, inputs, args, dtypes.clone()) {
            Ok(()) => match get_server_error(&client) {
                Some(e) => e,
                None => TestOutcome::Validated(assert_result(
                    &lhs_data,
                    &rhs_data,
                    &problem_for_check,
                    &client,
                    out,
                    dtypes,
                )),
            },
            Err(e) => TestOutcome::CompileError(format!("{e:?}")),
        };

    outcome.enforce()
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
