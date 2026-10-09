//! Force the split weight-gradient kernel and compare with the defining sum.

use cubecl::{prelude::*, std::tensor::TensorHandle};
use cubek_convolution::{
    ConvolutionArgs, DepthwiseWeightBackwardTensors, launch_depthwise_weight_backward,
};

fn upload(
    client: &Client,
    shape: [usize; 4],
    values: &[f32],
    channel_first: bool,
    gap: usize,
) -> TensorHandle {
    let order = if channel_first {
        [0, 3, 1, 2]
    } else {
        [0, 1, 2, 3]
    };
    let mut strides = [0; 4];
    let mut size = gap;
    for &axis in order.iter().rev() {
        strides[axis] = size;
        size *= shape[axis];
    }
    let mut data = vec![99.0f32; size];
    for (i, &value) in values.iter().enumerate() {
        let mut rem = i;
        let mut offset = 0;
        for axis in (0..4).rev() {
            offset += (rem % shape[axis]) * strides[axis];
            rem /= shape[axis];
        }
        data[offset] = value;
    }
    TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&data)),
        shape,
        strides,
        f32::elem_type_native(),
    )
}

#[test]
fn depthwise_weight_split_matches_reference() {
    let client = cubecl::test_device().client();
    check_depthwise_weight_reference(&client);
}

pub fn check_depthwise_weight_reference(client: &Client) {
    if client.properties().hardware.max_units_per_cube < 128 {
        return;
    }
    // shape NHWC, kernel, stride, dilation, beginning/end padding, multiplier.
    for (shape, kernel, stride, dilation, padding, end_padding, multiplier) in [
        ([1, 1, 1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], 1),
        ([2, 7, 9, 5], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], 1),
        ([3, 9, 11, 3], [2, 3], [2, 1], [2, 1], [0, 2], [1, 0], 2),
        ([2, 5, 7, 4], [3, 2], [2, 2], [2, 2], [3, 0], [0, 2], 1),
        // Exactly one full split, followed below by a partial second split.
        ([2, 32, 64, 3], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], 1),
        // More than one 4096-sample split, with a partial final split.
        ([4, 33, 35, 7], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], 1),
        // NHWC channel tiles: partial channel and spatial tiles, then a
        // multiplier with stride and dilation. Other layouts take scalar tiles.
        ([2, 17, 19, 35], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], 1),
        ([2, 19, 21, 33], [2, 3], [2, 1], [2, 2], [1, 0], [0, 2], 2),
    ] {
        let [batch, height, width, channels] = shape;
        let [kh, kw] = kernel;
        let oh =
            (height + padding[0] + end_padding[0] - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
        let ow = (width + padding[1] + end_padding[1] - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
        let out_channels = channels * multiplier;
        let grad_shape = [batch, oh, ow, out_channels];
        let values = (0..shape.iter().product())
            .map(|i| ((i * 7 % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let gradients = (0..grad_shape.iter().product())
            .map(|i| ((i * 3 % 13) as f32 - 6.0) / 16.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; out_channels * kh * kw];
        for n in 0..batch {
            for y in 0..oh {
                for x in 0..ow {
                    for oc in 0..out_channels {
                        let dy = gradients[((n * oh + y) * ow + x) * out_channels + oc];
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (y * stride[0] + ky * dilation[0]) as isize
                                    - padding[0] as isize;
                                let ix = (x * stride[1] + kx * dilation[1]) as isize
                                    - padding[1] as isize;
                                if iy >= 0 && ix >= 0 && iy < height as isize && ix < width as isize
                                {
                                    let index = ((n * height + iy as usize) * width + ix as usize)
                                        * channels
                                        + oc / multiplier;
                                    expected[(oc * kh + ky) * kw + kx] += values[index] * dy;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (input_nchw, grad_nchw, gap) in [(false, false, 1), (true, true, 1), (false, true, 2)] {
            let input = upload(client, shape, &values, input_nchw, gap);
            let out_grad = upload(client, grad_shape, &gradients, grad_nchw, gap);
            // The mixed-layout case also writes a permuted, gapped output.
            let weight_strides = if gap == 1 {
                [kh * kw, kw, 1, 1]
            } else {
                [gap, kw * out_channels * gap, out_channels * gap, 1]
            };
            let mut expected_storage = vec![99.0f32; expected.len() * gap];
            for oc in 0..out_channels {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let offset = oc * weight_strides[0]
                            + ky * weight_strides[1]
                            + kx * weight_strides[2];
                        expected_storage[offset] = expected[(oc * kh + ky) * kw + kx];
                    }
                }
            }
            let weight_grad = TensorHandle::new(
                client.create_from_slice(f32::as_bytes(&vec![99.0; expected_storage.len()])),
                [out_channels, kh, kw, 1],
                weight_strides,
                f32::elem_type_native(),
            );
            launch_depthwise_weight_backward(
                client,
                DepthwiseWeightBackwardTensors {
                    input: input.binding(),
                    out_grad: out_grad.binding(),
                    weight_grad: weight_grad.clone().binding(),
                },
                ConvolutionArgs {
                    stride,
                    padding,
                    dilation,
                },
                channels,
                f32::elem_type_native(),
            )
            .unwrap();
            let bytes = client.read_one_unchecked(weight_grad.handle);
            assert_eq!(
                f32::from_bytes(&bytes),
                &expected_storage,
                "shape={shape:?}, multiplier={multiplier}, input_nchw={input_nchw}, grad_nchw={grad_nchw}, gap={gap}"
            );
        }
    }
}

#[test]
fn depthwise_weight_declines_dense_and_other_dtypes() {
    let client = cubecl::test_device().client();
    check_depthwise_weight_declines(&client);
}

pub fn check_depthwise_weight_declines(client: &Client) {
    for (groups, dtype) in [
        (1, f32::elem_type_native()),
        (4, half::f16::elem_type_native()),
    ] {
        let input = TensorHandle::empty(client, [1, 5, 5, 4], dtype);
        let out_grad = TensorHandle::empty(client, [1, 3, 3, 4], dtype);
        let weight_grad = TensorHandle::empty(client, [4, 3, 3, 1], dtype);
        assert!(
            launch_depthwise_weight_backward(
                client,
                DepthwiseWeightBackwardTensors {
                    input: input.binding(),
                    out_grad: out_grad.binding(),
                    weight_grad: weight_grad.binding(),
                },
                ConvolutionArgs {
                    stride: [1, 1],
                    padding: [0, 0],
                    dilation: [1, 1]
                },
                groups,
                dtype,
            )
            .is_err()
        );
    }
}
