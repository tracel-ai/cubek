use cubecl::{TestRuntime, ir::AddressType, prelude::*, zspace::shape};
use cubek_convolution::{
    ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    definition::{BackwardWeightBlueprint, ConvBlueprint},
    launch_ref,
};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::multi_level::{
    components::tile::TileMatmulKind, definition::BatchMatmulBlueprint,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, launch_and_capture_outcome,
};

use super::common::{default_partition_buffering, default_swizzle, default_tiling_scheme};

/// End padding is encoded by the out-gradient extent. With input length 8, a three-tap kernel,
/// no beginning padding, and end padding 2, the output length is 8. The three weight taps then
/// overlap 8, 7, and 6 real input positions respectively.
#[test]
fn backward_weight_supports_end_only_padding() {
    let client = TestRuntime::client(&Default::default());
    let dtypes = MatmulElems::from_single_dtype(half::f16::elem_type_native());

    let batches = 1;
    let in_len = 8;
    let out_len = 8;
    let kernel_len = 3;
    let channels = 16;
    let out_channels = 16;

    let input = TestInput::builder(client.clone(), shape![batches, in_len, channels])
        .dtype(dtypes.lhs_global)
        .custom(vec![1.; batches * in_len * channels])
        .generate_without_host_data();
    let out_grad = TestInput::builder(client.clone(), shape![batches, out_len, out_channels])
        .dtype(dtypes.rhs_global)
        .custom(vec![1.; batches * out_len * out_channels])
        .generate_without_host_data();
    let weight_grad =
        TestInput::builder(client.clone(), shape![out_channels, kernel_len, channels])
            .dtype(dtypes.acc_global)
            .zeros()
            .generate_without_host_data();

    let problem = ConvolutionProblem {
        m: out_channels,
        n: channels * kernel_len,
        k: batches * out_len,
        lhs_strides: input.strides().clone(),
        rhs_strides: out_grad.strides().clone(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: vec![kernel_len as u32],
        stride: vec![1],
        padding: vec![0],
        dilation: vec![1],
        batches,
        channels,
        out_channels,
        in_shape: shape![in_len],
        out_shape: shape![out_len],
        padded_channels: channels,
        operation: ConvolutionOperation::BackwardWeight,
        dimensionality: Dimensionality::Dim1,
        global_dtypes: dtypes.as_global_elems(),
        address_type: AddressType::U32,
    };
    let matmul = BatchMatmulBlueprint::builder(
        TileMatmulKind::Cmma,
        default_tiling_scheme(),
        client.properties().hardware.plane_size_max,
        &problem.as_matmul_problem(),
    )
    .shared_swizzle(default_swizzle())
    .partition_buffering(default_partition_buffering())
    .build();

    let inputs = ConvolutionInputs::BackwardWeight {
        input: InputBinding::new(input.binding(), dtypes.lhs_global),
        out_grad: InputBinding::new(out_grad.binding(), dtypes.rhs_global),
        weight_grad: weight_grad.clone().binding(),
    };
    let args = ConvolutionArgs::<1> {
        stride: [1],
        padding: [0],
        dilation: [1],
    };
    let strategy = Strategy::Forced {
        algorithm: ConvAlgorithm::SimpleSyncCyclic,
        blueprint: ConvBlueprint::BackwardWeight(BackwardWeightBlueprint {
            matmul,
            dimensionality: Dimensionality::Dim1,
        }),
    };

    let outcome = launch_and_capture_outcome(&client, &[&weight_grad.handle], |client| {
        match launch_ref(&strategy, client, inputs, args, dtypes) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
        }
    });

    if let ExecutionOutcome::CompileError(error) = outcome {
        TestOutcome::CompileError(error).enforce();
        return;
    }

    let actual = HostData::from_tensor_handle(&client, weight_grad, HostDataType::F32);
    let expected_per_tap = [8., 7., 6.];

    for out_channel in 0..out_channels {
        for (tap, expected) in expected_per_tap.into_iter().enumerate() {
            for channel in 0..channels {
                assert_eq!(actual.get_f32(&[out_channel, tap, channel]), expected);
            }
        }
    }
}
