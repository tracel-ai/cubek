use cubecl::{
    ir::{AddressType, OpaqueType},
    prelude::*,
    zspace::shape,
};
use cubek_convolution::{
    ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    definition::{BackwardWeightBlueprint, ConvBlueprint},
    launch_ref,
};
use cubek_matmul::definition::{AccumulatorOperand, MatmulElems};
use cubek_matmul::multi_level::{
    PartitionSize, StageSize,
    components::{stage::PartitionBuffering, tile::TileMatmulKind},
    definition::{BatchMatmulBlueprint, TilingScheme},
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, launch_and_capture_outcome,
};

use super::common::{
    default_partition_buffering, default_swizzle, default_tile_size, default_tiling_scheme,
    f16_dtypes,
};

/// End padding is encoded by the out-gradient extent. With input length 8, a three-tap kernel,
/// no beginning padding, and end padding 2, the output length is 8. The three weight taps then
/// overlap 8, 7, and 6 real input positions respectively.
#[test]
fn backward_weight_supports_end_only_padding() {
    let client = cubecl::test_device().client();
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
        &problem.as_matmul_problem(AccumulatorOperand::Absent),
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

/// A 1x1 convolution on a 1x1 map, 16 to 256 channels, at batch 10. Its weight gradient
/// contracts over the batch alone: the matmul is `m = 256`, `n = 16`, `k = 10`.
const BATCHES: usize = 10;
const CHANNELS: usize = 16;
const OUT_CHANNELS: usize = 256;

/// Runs `algorithm` on the 1x1 weight gradient under a forced tiling, in f16, and checks every
/// value of the gradient.
fn check_1x1_weight_gradient(
    client: &Client,
    algorithm: ConvAlgorithm,
    tile_matmul: TileMatmulKind,
    tiling_scheme: TilingScheme,
) {
    let dtypes = f16_dtypes();
    let plane_dim = client.properties().hardware.plane_size_max;

    // Multiples of 1/8 below 1 in magnitude: f16 holds them, their products and these sums.
    let input_values: Vec<f32> = (0..BATCHES * CHANNELS)
        .map(|index| ((index % 17) as f32 - 8.) / 8.)
        .collect();
    let out_grad_values: Vec<f32> = (0..BATCHES * OUT_CHANNELS)
        .map(|index| (((index * 3 + 1) % 17) as f32 - 8.) / 8.)
        .collect();

    let input = TestInput::builder(client.clone(), shape![BATCHES, 1, 1, CHANNELS])
        .dtype(dtypes.lhs_global)
        .custom(input_values.clone())
        .generate_without_host_data();
    let out_grad = TestInput::builder(client.clone(), shape![BATCHES, 1, 1, OUT_CHANNELS])
        .dtype(dtypes.rhs_global)
        .custom(out_grad_values.clone())
        .generate_without_host_data();
    let weight_grad = TestInput::builder(client.clone(), shape![OUT_CHANNELS, 1, 1, CHANNELS])
        .dtype(dtypes.acc_global)
        .zeros()
        .generate_without_host_data();

    let problem = ConvolutionProblem {
        m: OUT_CHANNELS,
        n: CHANNELS,
        k: BATCHES,
        lhs_strides: input.strides().clone(),
        rhs_strides: out_grad.strides().clone(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: vec![1, 1],
        stride: vec![1, 1],
        padding: vec![0, 0],
        dilation: vec![1, 1],
        batches: BATCHES,
        channels: CHANNELS,
        out_channels: OUT_CHANNELS,
        in_shape: shape![1, 1],
        out_shape: shape![1, 1],
        padded_channels: CHANNELS,
        operation: ConvolutionOperation::BackwardWeight,
        dimensionality: Dimensionality::Dim2,
        global_dtypes: dtypes.as_global_elems(),
        address_type: AddressType::U32,
    };
    let matmul = BatchMatmulBlueprint::builder(
        tile_matmul,
        tiling_scheme,
        plane_dim,
        &problem.as_matmul_problem(AccumulatorOperand::Absent),
    )
    .partition_buffering(PartitionBuffering::Single)
    .build();
    let strategy = Strategy::Forced {
        algorithm,
        blueprint: ConvBlueprint::BackwardWeight(BackwardWeightBlueprint {
            matmul,
            dimensionality: Dimensionality::Dim2,
        }),
    };

    let inputs = ConvolutionInputs::BackwardWeight {
        input: InputBinding::new(input.binding(), dtypes.lhs_global),
        out_grad: InputBinding::new(out_grad.binding(), dtypes.rhs_global),
        weight_grad: weight_grad.clone().binding(),
    };
    let args = ConvolutionArgs::<2> {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    };

    let outcome = launch_and_capture_outcome(client, &[&weight_grad.handle], |client| {
        match launch_ref(&strategy, client, inputs, args, dtypes) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
        }
    });
    assert!(
        matches!(outcome, ExecutionOutcome::Executed),
        "{algorithm} declined {tiling_scheme:?}: {outcome:?}"
    );

    let actual = HostData::from_tensor_handle(client, weight_grad, HostDataType::F32);
    for out_channel in 0..OUT_CHANNELS {
        for channel in 0..CHANNELS {
            let expected: f32 = (0..BATCHES)
                .map(|batch| {
                    out_grad_values[batch * OUT_CHANNELS + out_channel]
                        * input_values[batch * CHANNELS + channel]
                })
                .sum();
            assert_eq!(
                actual.get_f32(&[out_channel, 0, 0, channel]),
                expected,
                "weight gradient at out channel {out_channel}, channel {channel}"
            );
        }
    }
}

/// Skips the test with `reason`.
fn skip(reason: &str) {
    TestOutcome::Validated(cubek_test_utils::ValidationResult::Skipped(
        reason.to_string(),
    ))
    .enforce();
}

/// Runs a strided reader on a stage its loading units do not divide evenly, and checks the weight
/// gradient it computes. Regression for the strided readers refusing such a stage: it must load,
/// and give the right gradient.
///
/// The blueprint holds one 16x8x16 f16 tile per stage, so the rhs stage is 16 vectors of 8 halves
/// for the 32 units of a plane.
fn strided_reader_loads_an_unevenly_split_stage(algorithm: ConvAlgorithm) {
    let client = cubecl::test_device().client();
    let dtypes = f16_dtypes();
    let plane_dim = client.properties().hardware.plane_size_max;
    let features = &client.properties().features;

    let offered = features.matmul.mma.iter().any(|cfg| {
        cfg.a_type == dtypes.lhs_register
            && cfg.b_type == dtypes.rhs_register
            && cfg.cd_type == dtypes.acc_register
            && (cfg.m, cfg.n, cfg.k) == (16, 8, 16)
    });
    let async_copy = matches!(
        algorithm,
        ConvAlgorithm::SimpleAsyncStrided | ConvAlgorithm::SpecializedAsyncStrided
    );
    if !offered {
        return skip("the device offers no 16x8x16 f16 mma");
    }
    if async_copy && !(features.copy_async && features.supports_type(OpaqueType::Barrier)) {
        return skip("the device has no async copy");
    }
    // The rhs stage is one k-by-n tile, read in 128-bit vectors.
    let stage_vectors = 16 * 8 * dtypes.rhs_stage.size() as u32 / 16;
    assert!(
        !stage_vectors.is_multiple_of(plane_dim),
        "{stage_vectors} vectors split evenly over {plane_dim} units: the test would not \
         exercise an uneven split"
    );

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((16, 8, 16).into())
        .with_partition_size(PartitionSize { m: 1, n: 1, k: 1 })
        .with_stage_size(StageSize { m: 1, n: 1, k: 1 })
        .build()
        .unwrap();
    check_1x1_weight_gradient(&client, algorithm, TileMatmulKind::Mma, tiling_scheme);
}

#[test]
fn simple_sync_strided_loads_an_unevenly_split_stage() {
    strided_reader_loads_an_unevenly_split_stage(ConvAlgorithm::SimpleSyncStrided);
}

#[test]
fn simple_async_strided_loads_an_unevenly_split_stage() {
    strided_reader_loads_an_unevenly_split_stage(ConvAlgorithm::SimpleAsyncStrided);
}

#[test]
fn specialized_async_strided_loads_an_unevenly_split_stage() {
    strided_reader_loads_an_unevenly_split_stage(ConvAlgorithm::SpecializedAsyncStrided);
}

/// A stage wider than the weight gradient's `n` writes only the gradient's own channels. Regression
/// for the transposed layout checking `n` with the `m` bounds flag, which zeroed the gradient.
#[test]
fn weight_gradient_stage_wider_than_its_channels_writes_only_its_channels() {
    let client = cubecl::test_device().client();
    let dtypes = f16_dtypes();
    let tile = default_tile_size();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == dtypes.lhs_register
            && cfg.b_type == dtypes.rhs_register
            && cfg.cd_type == dtypes.acc_register
            && (cfg.m, cfg.n, cfg.k) == (tile.m, tile.n, tile.k)
    });
    if !offered {
        return skip("the device offers no f16 cmma of the default tile size");
    }

    let stage_n = 2 * CHANNELS as u32;
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile)
        .with_partition_size(PartitionSize {
            m: 1,
            n: (stage_n / tile.n) as u8,
            k: 1,
        })
        .with_stage_size(StageSize { m: 1, n: 1, k: 1 })
        .build()
        .unwrap();
    check_1x1_weight_gradient(
        &client,
        ConvAlgorithm::SimpleSyncCyclic,
        TileMatmulKind::Cmma,
        tiling_scheme,
    );
}
