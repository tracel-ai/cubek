use cubecl::{ir::AddressType, prelude::*, zspace::shape};
use cubek_convolution::{
    ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    definition::{BackwardDataBlueprint, ConvBlueprint},
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

#[test]
fn backward_data_masks_non_divisible_stride_coordinates() {
    run_backward_data_case(
        16,
        7,
        2,
        &[
            16., 16., 32., 16., 32., 16., 32., 16., 32., 16., 32., 16., 32., 16., 16., 0.,
        ],
    );
}

/// End padding is not present in `ConvolutionArgs`; it is encoded by the larger out-gradient
/// extent. For input length 8, a three-tap kernel, and no beginning padding, end padding 2 makes
/// the forward output (and therefore `out_grad`) length 8 rather than 6.
#[test]
fn backward_data_supports_end_only_padding() {
    run_backward_data_case(8, 8, 1, &[16., 32., 48., 48., 48., 48., 48., 48.]);
}

fn run_backward_data_case(
    in_len: usize,
    out_len: usize,
    stride: usize,
    expected_per_position: &[f32],
) {
    let client = cubecl::test_device().client();
    let dtypes = MatmulElems::from_single_dtype(half::f16::elem_type_native());

    let batches = 1;
    let kernel_len = 3;
    let channels = 16;
    let out_channels = 16;

    assert_eq!(expected_per_position.len(), in_len);

    let out_grad = TestInput::builder(client.clone(), shape![batches, out_len, out_channels])
        .dtype(dtypes.lhs_global)
        .custom(vec![1.; batches * out_len * out_channels])
        .generate_without_host_data();
    let weights = TestInput::builder(client.clone(), shape![out_channels, kernel_len, channels])
        .dtype(dtypes.rhs_global)
        .custom(vec![1.; out_channels * kernel_len * channels])
        .generate_without_host_data();
    let in_grad = TestInput::builder(client.clone(), shape![batches, in_len, channels])
        .dtype(dtypes.acc_global)
        .zeros()
        .generate_without_host_data();

    let problem = ConvolutionProblem {
        m: batches * in_len,
        n: channels,
        k: out_channels * kernel_len,
        lhs_strides: out_grad.strides().clone(),
        rhs_strides: weights.strides().clone(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: vec![kernel_len as u32],
        stride: vec![stride as u32],
        padding: vec![0],
        dilation: vec![1],
        batches,
        channels,
        out_channels,
        in_shape: shape![in_len],
        out_shape: shape![out_len],
        padded_channels: out_channels,
        operation: ConvolutionOperation::BackwardData,
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

    let inputs = ConvolutionInputs::BackwardData {
        out_grad: InputBinding::new(out_grad.binding(), dtypes.lhs_global),
        weights: InputBinding::new(weights.binding(), dtypes.rhs_global),
        in_grad: in_grad.clone().binding(),
    };
    let args = ConvolutionArgs::<1> {
        stride: [stride],
        padding: [0],
        dilation: [1],
    };
    let strategy = Strategy::Forced {
        algorithm: ConvAlgorithm::SimpleSyncCyclic,
        blueprint: ConvBlueprint::BackwardData(BackwardDataBlueprint {
            matmul,
            dimensionality: Dimensionality::Dim1,
        }),
    };

    let outcome =
        launch_and_capture_outcome(&client, &[&in_grad.handle], |client| {
            match launch_ref(&strategy, client, inputs, args, dtypes) {
                Ok(()) => ExecutionOutcome::Executed,
                Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
            }
        });

    if let ExecutionOutcome::CompileError(error) = outcome {
        TestOutcome::CompileError(error).enforce();
        return;
    }

    let actual = HostData::from_tensor_handle(&client, in_grad, HostDataType::F32);

    for (position, &expected) in expected_per_position.iter().enumerate() {
        for channel in 0..channels {
            assert_eq!(actual.get_f32(&[0, position, channel]), expected);
        }
    }
}
