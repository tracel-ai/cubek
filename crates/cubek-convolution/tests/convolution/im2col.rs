//! Check explicit lowering independently of any convolution or matmul implementation.

use cubecl::{ir::FloatKind, prelude::*, std::tensor::TensorHandle};
use cubek_convolution::{ConvolutionArgs, Im2colTensors, launch_im2col};

fn upload(
    client: &Client,
    shape: &[usize],
    values: &[f32],
    channel_first: bool,
    gap: usize,
) -> TensorHandle {
    let rank = shape.len();
    let mut order = vec![0];
    if channel_first {
        order.push(rank - 1);
        order.extend(1..rank - 1);
    } else {
        order.extend(1..rank);
    }
    let mut strides = vec![0; rank];
    let mut size = gap;
    for &axis in order.iter().rev() {
        strides[axis] = size;
        size *= shape[axis];
        if axis == rank - 1 && !channel_first {
            size = size.next_multiple_of(8);
        }
    }
    let mut data = vec![99.0; size];
    for (i, value) in values.iter().enumerate() {
        let mut rem = i;
        let mut offset = 0;
        for axis in (0..rank).rev() {
            offset += rem % shape[axis] * strides[axis];
            rem /= shape[axis];
        }
        data[offset] = *value;
    }
    TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&data)),
        shape.to_vec(),
        strides,
        f32::elem_type_native(),
    )
}

fn check<const N: usize>(
    client: &Client,
    batch: usize,
    spatial: [usize; N],
    channels: usize,
    kernel: [usize; N],
    stride: [usize; N],
    dilation: [usize; N],
    padding: [(usize, usize); N],
) {
    let out_shape: [usize; N] = core::array::from_fn(|i| {
        (spatial[i] + padding[i].0 + padding[i].1 - dilation[i] * (kernel[i] - 1) - 1) / stride[i]
            + 1
    });
    let mut shape = vec![batch];
    shape.extend(spatial);
    shape.push(channels);
    let values = (0..shape.iter().product())
        .map(|i| ((i * 7 % 31) as f32 - 15.0) / 8.0)
        .collect::<Vec<_>>();
    let rows = batch * out_shape.iter().product::<usize>();
    let width = channels * kernel.iter().product::<usize>();
    for (channel_first, input_gap, output_gap) in [(false, 1, 1), (true, 1, 1), (false, 2, 2)] {
        let row_stride = (width * output_gap + 1).next_multiple_of(16);
        let mut expected = vec![99.0; rows * row_stride];
        for row in 0..rows {
            let mut b = row;
            let mut out_pos = [0; N];
            for axis in (0..N).rev() {
                out_pos[axis] = b % out_shape[axis];
                b /= out_shape[axis];
            }
            for col in 0..width {
                let channel = col % channels;
                let mut tap = col / channels;
                let mut coords = [0; N];
                let mut valid = true;
                for axis in (0..N).rev() {
                    let k = tap % kernel[axis];
                    tap /= kernel[axis];
                    let p = (out_pos[axis] * stride[axis] + k * dilation[axis]) as isize
                        - padding[axis].0 as isize;
                    valid &= p >= 0 && p < spatial[axis] as isize;
                    coords[axis] = p.max(0) as usize;
                }
                let value = if valid {
                    let mut index = b;
                    for axis in 0..N {
                        index = index * spatial[axis] + coords[axis];
                    }
                    values[index * channels + channel]
                } else {
                    0.0
                };
                expected[row * row_stride + col * output_gap] = value;
            }
        }
        let input = upload(client, &shape, &values, channel_first, input_gap);
        let columns = TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&vec![99.0; expected.len()])),
            [rows, width],
            [row_stride, output_gap],
            f32::elem_type_native(),
        );
        launch_im2col(
            client,
            Im2colTensors {
                input: input.binding(),
                columns: columns.clone().binding(),
            },
            out_shape,
            kernel,
            ConvolutionArgs {
                stride,
                padding: padding.map(|p| p.0),
                dilation,
            },
            f32::elem_type_native(),
        )
        .unwrap();
        let bytes = client.read_one_unchecked(columns.handle);
        assert_eq!(
            f32::from_bytes(&bytes),
            expected,
            "N={N}, channels={channels}, channel_first={channel_first}, gap={input_gap}"
        );
    }
}

pub fn check_im2col_reference(client: &Client) {
    check(client, 2, [13], 3, [3], [2], [2], [(1, 2)]);
    check(
        client,
        1,
        [1, 1],
        4,
        [3, 3],
        [1, 1],
        [1, 1],
        [(2, 1), (1, 2)],
    );
    check(
        client,
        2,
        [13, 17],
        4,
        [3, 3],
        [1, 1],
        [1, 1],
        [(1, 1), (1, 1)],
    );
    check(
        client,
        3,
        [9, 11],
        17,
        [2, 3],
        [2, 1],
        [2, 2],
        [(0, 1), (3, 0)],
    );
    check(
        client,
        2,
        [5, 7],
        32,
        [3, 2],
        [2, 2],
        [1, 2],
        [(1, 0), (0, 2)],
    );
    check(
        client,
        1,
        [4, 5, 7],
        4,
        [2, 3, 2],
        [1, 2, 1],
        [2, 1, 2],
        [(1, 0), (0, 1), (2, 0)],
    );
}

pub fn check_im2col_declines(client: &Client) {
    let dtype = ElemType::Float(FloatKind::F16);
    let input = TensorHandle::empty(client, [1, 5, 5, 4], dtype);
    let columns = TensorHandle::empty(client, [25, 36], dtype);
    assert!(
        launch_im2col(
            client,
            Im2colTensors {
                input: input.binding(),
                columns: columns.binding()
            },
            [5, 5],
            [3, 3],
            ConvolutionArgs {
                stride: [1, 1],
                padding: [1, 1],
                dilation: [1, 1]
            },
            dtype,
        )
        .is_err()
    );
    for stride in [[0, 1], [1, 1]] {
        let dtype = f32::elem_type_native();
        let input = TensorHandle::empty(client, [1, 5, 5, 4], dtype);
        let columns = TensorHandle::empty(client, [25, 35], dtype);
        assert!(
            launch_im2col(
                client,
                Im2colTensors {
                    input: input.binding(),
                    columns: columns.binding()
                },
                [5, 5],
                [3, 3],
                ConvolutionArgs {
                    stride,
                    padding: [1, 1],
                    dilation: [1, 1]
                },
                dtype,
            )
            .is_err()
        );
    }
}

#[test]
fn im2col_matches_reference() {
    check_im2col_reference(&cubecl::test_device().client());
}

#[test]
fn im2col_declines_unsupported_inputs() {
    check_im2col_declines(&cubecl::test_device().client());
}
