use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_convolution::{
    AcceleratedTileKind, ConvolutionArgs,
    backward_weight::split::{AcceleratedSplitStrategy, PartialColumnOrder, launch_accelerated},
    components::ConvSetupError,
};
use cubek_matmul::definition::MatmulElems;
use cubek_std::InputBinding;
use cubek_test_utils::{ExecutionOutcome, TestOutcome, launch_and_capture_outcome};

use super::unit::{assert_close, binding, upload, values};

#[test]
fn accelerated_split_weight_gradient_keeps_f16_partials_in_f32() {
    check_reduced_precision::<half::f16>();
}

#[test]
fn accelerated_split_weight_gradient_preserves_bf16() {
    check_reduced_precision::<half::bf16>();
}

// Reads inputs [2, 512, 4] and gradients [2, 512, 8], and checks weights [8, 1, 4].
fn check_reduced_precision<E: CubeElement + Numeric + PartialEq>() {
    let client = cubecl::test_device().client();
    let dtype = E::elem_type_native();
    let mut input_values = [
        vec![E::from_int(200); 512 * 4],
        vec![E::from_int(-200); 512 * 4],
    ]
    .concat();
    for index in (512 * 4 + 3..input_values.len()).step_by(4) {
        input_values[index] = E::from_int(-199);
    }
    let grad_values = vec![E::from_int(1); 2 * 512 * 8];
    let input = TensorHandle::new(
        client.create_from_slice(E::as_bytes(&input_values)),
        [2, 512, 4],
        [512 * 4, 4, 1],
        dtype,
    );
    let grad = TensorHandle::new(
        client.create_from_slice(E::as_bytes(&grad_values)),
        [2, 512, 8],
        [512 * 8, 8, 1],
        dtype,
    );

    for column_order in [
        PartialColumnOrder::ChannelsLast,
        PartialColumnOrder::SpatialLast,
    ] {
        let output = TensorHandle::empty(&client, [8, 1, 4], dtype);
        let outcome = launch_and_capture_outcome(&client, &[&output.handle], |client| {
            let args = ConvolutionArgs {
                stride: [1],
                padding: [0],
                dilation: [1],
            };
            let input = InputBinding::new(input.clone().binding(), dtype);
            let grad = InputBinding::new(grad.clone().binding(), dtype);
            let result = launch_accelerated::<1>(
                client,
                input,
                grad,
                output.clone().binding(),
                args,
                AcceleratedSplitStrategy {
                    tile_kind: AcceleratedTileKind::Mma,
                    column_order,
                },
                MatmulElems::from_single_dtype(dtype),
            );
            match result {
                Ok(()) => ExecutionOutcome::Executed,
                Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
            }
        });
        if let ExecutionOutcome::CompileError(error) = outcome {
            TestOutcome::CompileError(error).enforce();
            return;
        }
        let strides = output.strides().to_vec();
        let bytes = client.read_one_unchecked(output.handle);
        let actual = E::from_bytes(&bytes);
        for co in 0..8 {
            for ci in 0..4 {
                let expected = if ci == 3 { 512 } else { 0 };
                assert_eq!(
                    actual[co * strides[0] + ci * strides[2]],
                    E::from_int(expected)
                );
            }
        }
    }
}

#[test]
fn accelerated_split_weight_gradient_handles_tail_and_output_strides() {
    check_weight_gradient(
        &[8, 1031, 3],
        [3],
        ConvolutionArgs {
            stride: [2],
            padding: [1],
            dilation: [2],
        },
        7,
    );
}

#[test]
fn accelerated_split_weight_gradient_supports_three_spatial_axes() {
    check_weight_gradient(
        &[16, 5, 7, 9, 4],
        [3, 3, 3],
        ConvolutionArgs {
            stride: [1; 3],
            padding: [0; 3],
            dilation: [1; 3],
        },
        8,
    );
}

#[test]
fn accelerated_split_weight_gradient_handles_vector_alignment() {
    for width in [20, 19] {
        for (stride, padding, dilation) in [
            ([2, 4], [0, 0], [1; 2]),
            ([2, 4], [1, 4], [1; 2]),
            ([2, 4], [1, 1], [1; 2]),
            ([2, 3], [0, 0], [1; 2]),
            ([2, 4], [0, 0], [2; 2]),
        ] {
            check_weight_gradient(
                &[32, 16, width, 4],
                [4, 8],
                ConvolutionArgs {
                    stride,
                    padding,
                    dilation,
                },
                8,
            );
        }
    }
}

#[test]
fn accelerated_split_weight_gradient_rejects_wide_source_coordinates() {
    let client = cubecl::test_device().client();
    let input = upload(&client, &[1, 1, 1, 4], &[1.0; 4], false);
    let grad = upload(&client, &[1, 1, 1, 8], &[1.0; 8], false);
    let output = upload(&client, &[8, 3, 3, 4], &[0.0; 288], false);
    let result = launch_accelerated::<2>(
        &client,
        binding(&input),
        binding(&grad),
        output.binding(),
        ConvolutionArgs {
            stride: [1; 2],
            padding: [0; 2],
            dilation: [1usize << 31; 2],
        },
        AcceleratedSplitStrategy {
            tile_kind: AcceleratedTileKind::Mma,
            column_order: PartialColumnOrder::ChannelsLast,
        },
        MatmulElems::from_single_dtype(f32::elem_type_native()),
    );
    assert!(matches!(result, Err(ConvSetupError::InvalidConfig(_))));
}

// Checks weights [out_channels, ...kernel, channels] from rank N + 2 inputs
// [batch, ...spatial, channels] and gradients [batch, ...output_spatial, out_channels].
fn check_weight_gradient<const N: usize>(
    shape: &[usize],
    kernel: [usize; N],
    args: ConvolutionArgs<N>,
    out_channels: usize,
) {
    let client = cubecl::test_device().client();
    let channels = shape[N + 1];
    let out_spatial: [usize; N] = core::array::from_fn(|axis| {
        (shape[axis + 1] + 2 * args.padding[axis] - args.dilation[axis] * (kernel[axis] - 1) - 1)
            / args.stride[axis]
            + 1
    });
    let grad_shape = [vec![shape[0]], out_spatial.to_vec(), vec![out_channels]].concat();
    let weight_shape = [vec![out_channels], kernel.to_vec(), vec![channels]].concat();
    let taps = kernel.iter().product::<usize>();
    let input = values(shape.iter().product(), 3);
    let grad = values(grad_shape.iter().product(), 41);
    let mut expected = vec![0.0_f64; weight_shape.iter().product()];

    for (index, gradient) in grad.iter().enumerate() {
        let co = index % out_channels;
        let mut batch = index / out_channels;
        let mut coords = [0; N];
        for axis in (0..N).rev() {
            coords[axis] = batch % out_spatial[axis];
            batch /= out_spatial[axis];
        }
        for tap in 0..taps {
            let mut remaining = tap;
            let mut input_coords = [0; N];
            let mut valid = true;
            for axis in (0..N).rev() {
                let coordinate = (coords[axis] * args.stride[axis]
                    + remaining % kernel[axis] * args.dilation[axis])
                    as isize
                    - args.padding[axis] as isize;
                remaining /= kernel[axis];
                valid &= coordinate >= 0 && coordinate < shape[axis + 1] as isize;
                input_coords[axis] = coordinate.max(0) as usize;
            }
            if !valid {
                continue;
            }
            let mut pixel = batch;
            for axis in 0..N {
                pixel = pixel * shape[axis + 1] + input_coords[axis];
            }
            for ci in 0..channels {
                expected[(co * taps + tap) * channels + ci] +=
                    f64::from(*gradient) * f64::from(input[pixel * channels + ci]);
            }
        }
    }

    for channel_first in [false, true] {
        let input = upload_input(&client, shape, &input, channel_first);
        let grad = upload(&client, &grad_shape, &grad, channel_first);
        for column_order in [
            PartialColumnOrder::ChannelsLast,
            PartialColumnOrder::SpatialLast,
        ] {
            let output = upload(
                &client,
                &weight_shape,
                &vec![99.0; expected.len()],
                channel_first,
            );
            let outcome = launch_and_capture_outcome(&client, &[&output.handle], |client| {
                let result = launch_accelerated::<N>(
                    client,
                    binding(&input),
                    binding(&grad),
                    output.clone().binding(),
                    args.clone(),
                    AcceleratedSplitStrategy {
                        tile_kind: AcceleratedTileKind::Mma,
                        column_order,
                    },
                    MatmulElems::from_single_dtype(f32::elem_type_native()),
                );
                match result {
                    Ok(()) => ExecutionOutcome::Executed,
                    Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
                }
            });
            if let ExecutionOutcome::CompileError(error) = outcome {
                TestOutcome::CompileError(error).enforce();
                return;
            }
            assert_close(
                &client,
                output,
                &expected,
                "accelerated split weight gradient",
            );
        }
    }
}

// Stores [batch, ...spatial, channels] with a padded final spatial axis when channels come first.
fn upload_input(
    client: &Client,
    shape: &[usize],
    values: &[f32],
    channel_first: bool,
) -> TensorHandle {
    if !channel_first {
        return upload(client, shape, values, false);
    }

    let channels = shape[shape.len() - 1];
    let width = shape[shape.len() - 2];
    let padded_width = width.next_multiple_of(4);
    let mut stored_shape = shape.to_vec();
    stored_shape[shape.len() - 2] = padded_width;
    let mut stored = vec![99.0; stored_shape.iter().product()];
    for (index, value) in values.iter().enumerate() {
        let row = index / (width * channels);
        let x = index / channels % width;
        let channel = index % channels;
        stored[(row * padded_width + x) * channels + channel] = *value;
    }
    let tensor = upload(client, &stored_shape, &stored, true);
    let strides = tensor.strides().to_vec();
    TensorHandle::new(
        tensor.handle,
        shape.to_vec(),
        strides,
        f32::elem_type_native(),
    )
}
