//! Forced SIMT forward and gradient checks against an independent CPU convolution.

use cubecl::{prelude::*, std::tensor::TensorHandle};
use cubek_convolution::{
    ConvolutionArgs, ConvolutionInputs, UnitStrategy, UnitTiling, launch_unit,
};
use cubek_matmul::{definition::MatmulElems, multi_level::routines::TileSizeSelection};
use cubek_std::InputBinding;

fn upload(client: &Client, shape: &[usize], values: &[f32], channel_first: bool) -> TensorHandle {
    let rank = shape.len();
    let mut order = vec![0];
    if channel_first {
        order.push(rank - 1);
        order.extend(1..rank - 1);
    } else {
        order.extend(1..rank);
    }
    let mut strides = vec![0; rank];
    let mut len = 1;
    for &axis in order.iter().rev() {
        strides[axis] = len;
        len *= shape[axis];
        if axis == rank - 1 && !channel_first && rank > 1 {
            len = len.next_multiple_of(8);
        }
    }
    // Poison alignment gaps and a guard row. Missing channel/batch masks
    // must not accidentally pass because padding happened to contain zero.
    let mut data = vec![99.0; len + strides[0]];
    for (index, value) in values.iter().enumerate() {
        data[offset(index, shape, &strides)] = *value;
    }
    TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&data)),
        shape.to_vec(),
        strides,
        f32::elem_type_native(),
    )
}

fn offset(mut index: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0;
    for axis in (0..shape.len()).rev() {
        offset += (index % shape[axis]) * strides[axis];
        index /= shape[axis];
    }
    offset
}

fn assert_close(client: &Client, tensor: TensorHandle, expected: &[f64], context: &str) {
    let strides = tensor.strides().to_vec();
    let shape = tensor.shape().to_vec();
    let bytes = client.read_one_unchecked(tensor.handle);
    let values = f32::from_bytes(&bytes);
    for (i, expected) in expected.iter().enumerate() {
        let actual = values[offset(i, &shape, &strides)] as f64;
        assert!(
            (actual - expected).abs() <= 2e-4 * (1.0 + expected.abs()),
            "{context}: index {i}, actual {actual}, expected {expected}"
        );
    }
}

fn binding(tensor: &TensorHandle) -> InputBinding {
    InputBinding::new(tensor.clone().binding(), f32::elem_type_native())
}

fn values(size: usize, seed: usize) -> Vec<f32> {
    (0..size)
        .map(|i| (((i * 17 + seed) % 67) as f32 - 33.0) / 64.0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn check<const N: usize>(
    client: &Client,
    batch: usize,
    spatial: [usize; N],
    channels: usize,
    out_channels: usize,
    kernel: [usize; N],
    stride: [usize; N],
    dilation: [usize; N],
    padding: [(usize, usize); N],
) {
    let out_spatial: [usize; N] = core::array::from_fn(|i| {
        (spatial[i] + padding[i].0 + padding[i].1 - dilation[i] * (kernel[i] - 1) - 1) / stride[i]
            + 1
    });
    let shape = [vec![batch], spatial.to_vec(), vec![channels]].concat();
    let out_shape = [vec![batch], out_spatial.to_vec(), vec![out_channels]].concat();
    let weight_shape = [vec![out_channels], kernel.to_vec(), vec![channels]].concat();
    let x = values(shape.iter().product(), 3);
    let w = values(weight_shape.iter().product(), 23);
    let dy = values(out_shape.iter().product(), 41);
    let bias = values(out_channels, 53);
    let mut expected_y = vec![0.0f64; dy.len()];
    let mut expected_dx = vec![0.0f64; x.len()];
    let mut expected_dw = vec![0.0f64; w.len()];
    let taps = kernel.iter().product::<usize>();
    // Accumulate all three results by iterating the *forward* windows. This
    // also checks the inverse stride/padding logic used by the GPU dgrad.
    for (output_index, y) in expected_y.iter_mut().enumerate() {
        let co = output_index % out_channels;
        let mut b = output_index / out_channels;
        let mut coords = [0; N];
        for axis in (0..N).rev() {
            coords[axis] = b % out_spatial[axis];
            b /= out_spatial[axis];
        }
        for tap in 0..taps {
            let mut rem = tap;
            let mut input_coords = [0; N];
            let mut valid = true;
            for axis in (0..N).rev() {
                let k = rem % kernel[axis];
                rem /= kernel[axis];
                let p = (coords[axis] * stride[axis] + k * dilation[axis]) as isize
                    - padding[axis].0 as isize;
                valid &= p >= 0 && p < spatial[axis] as isize;
                input_coords[axis] = p.max(0) as usize;
            }
            if !valid {
                continue;
            }
            let mut pixel = b;
            for axis in 0..N {
                pixel = pixel * spatial[axis] + input_coords[axis];
            }
            for ci in 0..channels {
                let xi = pixel * channels + ci;
                let wi = (co * taps + tap) * channels + ci;
                *y += x[xi] as f64 * w[wi] as f64;
                expected_dx[xi] += dy[output_index] as f64 * w[wi] as f64;
                expected_dw[wi] += dy[output_index] as f64 * x[xi] as f64;
            }
        }
    }
    for channel_first in [false, true] {
        let input = upload(client, &shape, &x, channel_first);
        let weight = upload(client, &weight_shape, &w, channel_first);
        let out_grad = upload(client, &out_shape, &dy, channel_first);
        let bias_tensor = if channel_first {
            let data = bias.iter().flat_map(|x| [*x, 99.0]).collect::<Vec<_>>();
            TensorHandle::new(
                client.create_from_slice(f32::as_bytes(&data)),
                [out_channels],
                [2],
                f32::elem_type_native(),
            )
        } else {
            upload(client, &[out_channels], &bias, false)
        };
        for tile_size in [
            UnitStrategy::Inferred(TileSizeSelection::MinTileSize),
            UnitStrategy::Inferred(TileSizeSelection::MaxTileSize),
            UnitStrategy::Fixed(UnitTiling::NARROW),
            UnitStrategy::Fixed(UnitTiling::BALANCED),
            UnitStrategy::Fixed(UnitTiling::WIDE),
            UnitStrategy::Fixed(UnitTiling::DEEP),
            UnitStrategy::Fixed(UnitTiling::TALL),
        ] {
            for with_bias in [false, true] {
                let out = upload(client, &out_shape, &vec![99.0; dy.len()], channel_first);
                launch_unit(
                    tile_size,
                    client,
                    ConvolutionInputs::Forward {
                        input: binding(&input),
                        weight: binding(&weight),
                        bias: with_bias.then(|| binding(&bias_tensor)),
                        out: out.clone().binding(),
                    },
                    ConvolutionArgs {
                        stride,
                        padding: padding.map(|p| p.0),
                        dilation,
                    },
                    MatmulElems::new_deprecated::<f32>(),
                )
                .unwrap();
                let expected = expected_y
                    .iter()
                    .enumerate()
                    .map(|(i, y)| {
                        y + if with_bias {
                            bias[i % out_channels] as f64
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                assert_close(
                    client,
                    out,
                    &expected,
                    &format!(
                        "fwd N={N} c={channels} co={out_channels} {tile_size:?} bias={with_bias}"
                    ),
                );
            }
            let in_grad = upload(client, &shape, &vec![99.0; x.len()], channel_first);
            launch_unit(
                tile_size,
                client,
                ConvolutionInputs::BackwardData {
                    out_grad: binding(&out_grad),
                    weights: binding(&weight),
                    in_grad: in_grad.clone().binding(),
                },
                ConvolutionArgs {
                    stride,
                    padding: padding.map(|p| p.0),
                    dilation,
                },
                MatmulElems::new_deprecated::<f32>(),
            )
            .unwrap();
            assert_close(
                client,
                in_grad,
                &expected_dx,
                &format!("dgrad N={N} c={channels} co={out_channels} {tile_size:?}"),
            );
            let weight_grad = upload(client, &weight_shape, &vec![99.0; w.len()], channel_first);
            launch_unit(
                tile_size,
                client,
                ConvolutionInputs::BackwardWeight {
                    input: binding(&input),
                    out_grad: binding(&out_grad),
                    weight_grad: weight_grad.clone().binding(),
                },
                ConvolutionArgs {
                    stride,
                    padding: padding.map(|p| p.0),
                    dilation,
                },
                MatmulElems::new_deprecated::<f32>(),
            )
            .unwrap();
            assert_close(
                client,
                weight_grad,
                &expected_dw,
                &format!("wgrad N={N} c={channels} co={out_channels} {tile_size:?}"),
            );
        }
    }
}

pub fn check_unit_reference(client: &Client) {
    check_unit_channel_padding(client);
    check(client, 2, [13], 3, 7, [3], [2], [2], [(1, 2)]);
    check(
        client,
        2,
        [7, 9],
        4,
        8,
        [3, 3],
        [1, 1],
        [1, 1],
        [(1, 1), (1, 1)],
    );
    check(
        client,
        2,
        [9, 11],
        17,
        33,
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
        5,
        [2, 3, 2],
        [1, 2, 1],
        [2, 1, 2],
        [(1, 0), (0, 1), (2, 0)],
    );
    // Cross the split-K threshold with an uneven final interval and tail channels.
    check(
        client,
        2,
        [37, 61],
        3,
        7,
        [3, 3],
        [1, 1],
        [1, 1],
        [(1, 1), (1, 1)],
    );
    check(client, 1, [5003], 4, 8, [3], [1], [1], [(1, 1)]);
    check(
        client,
        1,
        [2, 17, 137],
        3,
        7,
        [1, 3, 2],
        [1, 1, 1],
        [1, 1, 1],
        [(0, 0), (1, 1), (1, 0)],
    );
    // Many partials and a short final split, as in large feature maps.
    check(
        client,
        4,
        [161, 163],
        4,
        4,
        [3, 3],
        [1, 1],
        [1, 1],
        [(1, 1), (1, 1)],
    );
}

pub fn check_unit_channel_padding(client: &Client) {
    // The backward-data contraction pads C_out, even when C_in already
    // equals the padded C_out. Comparing C_in to the padding misses this tail.
    check(
        client,
        2,
        [5, 7],
        8,
        7,
        [3, 3],
        [1, 1],
        [1, 1],
        [(1, 1), (1, 1)],
    );
    check(client, 1, [9], 4, 3, [3], [2], [1], [(1, 1)]);
}

#[test]
fn unit_convolution_matches_reference() {
    check_unit_reference(&cubecl::test_device().client());
}
