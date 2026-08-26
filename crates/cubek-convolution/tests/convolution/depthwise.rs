//! The depthwise routine against a CPU reference, over every tiling the catalogue offers.
//!
//! The tile suite next door (`cubek-tile/tests/tile/depthwise.rs`) checks that the *DSL* can
//! express a depthwise convolution, and builds its own space to do it. This checks the space
//! `launch_depthwise` actually builds — which is the one that ships, and the one whose tiling
//! moves whenever the kernel is optimized. A tiling that is fast and wrong has to fail here.

use cubecl::{Runtime, TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_convolution::{
    ConvolutionArgs, DepthwiseStrategy, DepthwiseTensors, DepthwiseTiling,
    components::ConvSetupError, launch_depthwise,
};
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// One depthwise convolution, in the layout the routine takes: NHWC in and out, `[C, kh, kw, 1]`
/// weights.
struct Case {
    b: usize,
    c: usize,
    /// The side of the input map.
    size: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Case {
    fn out_size(&self) -> usize {
        let reach = self.dilation * (self.k - 1) + 1;
        (self.size + 2 * self.padding - reach) / self.stride + 1
    }

    /// Small integers on a short period, so every accumulation is exact in `f32` and the
    /// comparison below can demand equality rather than an epsilon.
    fn ramp(n: usize, period: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
    }

    /// The definition, written out: no sum over input channels, and a tap outside the input
    /// contributes nothing.
    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let (size, c, out) = (self.size, self.c, self.out_size());
        let mut result = vec![0.0f32; self.b * out * out * c];

        for b in 0..self.b {
            for oh in 0..out {
                for ow in 0..out {
                    for ch in 0..c {
                        let mut acc = 0.0f32;
                        for rh in 0..self.k {
                            for rw in 0..self.k {
                                let h = (oh * self.stride + rh * self.dilation) as isize
                                    - self.padding as isize;
                                let w = (ow * self.stride + rw * self.dilation) as isize
                                    - self.padding as isize;
                                if h < 0 || w < 0 || h >= size as isize || w >= size as isize {
                                    continue;
                                }
                                let x =
                                    input[((b * size + h as usize) * size + w as usize) * c + ch];
                                // `[C, kh, kw, 1]`: the channel is outermost, not innermost.
                                let f = weight[(ch * self.k + rh) * self.k + rw];
                                acc += x * f;
                            }
                        }
                        result[((b * out + oh) * out + ow) * c + ch] = acc;
                    }
                }
            }
        }

        result
    }

    fn check(&self, tiling: DepthwiseTiling) -> Result<(), String> {
        self.check_with_weight_gap(tiling, 1)
    }

    /// Check a weight binding whose logical trailing singleton occupies one element out of every
    /// `weight_gap` in its backing buffer. A gap above one makes every non-singleton weight axis
    /// strided while preserving the logical `[C, kh, kw, 1]` shape.
    fn check_with_weight_gap(
        &self,
        tiling: DepthwiseTiling,
        weight_gap: usize,
    ) -> Result<(), String> {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        let dtype = f32::elem_type_native();
        let out_size = self.out_size();

        let in_shape = [self.b, self.size, self.size, self.c];
        let w_shape = [self.c, self.k, self.k, 1];
        let out_shape = [self.b, out_size, out_size, self.c];

        let in_data = Self::ramp(in_shape.iter().product(), 7);
        let w_data = Self::ramp(w_shape.iter().product(), 5);
        let physical_w_shape = [self.c, self.k, self.k, weight_gap];
        let mut physical_w_data = vec![99.0; physical_w_shape.iter().product()];
        for (physical, &logical) in physical_w_data
            .chunks_exact_mut(weight_gap)
            .zip(w_data.iter())
        {
            physical[0] = logical;
        }

        let (input, _) = TestInput::builder(client.clone(), Shape::new(in_shape))
            .dtype(dtype)
            .custom(in_data.clone())
            .generate_with_f32_host_data();
        let (weight, _) = TestInput::builder(client.clone(), Shape::new(physical_w_shape))
            .dtype(dtype)
            .custom(physical_w_data)
            .generate_with_f32_host_data();
        let mut weight = weight.binding();
        // Keep the physical binding's strides while exposing only its logical singleton axis.
        weight.shape = Shape::new(w_shape);
        // Zeroed, not empty: the kernel writes every cell it owns, so a cell left untouched by a
        // buggy walk shows up as the zero it started as rather than as whatever was there.
        let out: TensorHandle<TestRuntime> =
            TestInput::builder(client.clone(), Shape::new(out_shape))
                .dtype(dtype)
                .zeros()
                .generate_without_host_data();

        launch_depthwise::<TestRuntime>(
            &client,
            DepthwiseTensors {
                input: input.binding(),
                weight,
                out: out.clone().binding(),
            },
            ConvolutionArgs::<2> {
                stride: [self.stride; 2],
                padding: [self.padding; 2],
                dilation: [self.dilation; 2],
            },
            self.c,
            dtype,
            DepthwiseStrategy::Fixed(tiling),
        )
        .expect("the routine accepts a depthwise convolution");

        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let want = self.reference(&in_data, &w_data);

        for b in 0..self.b {
            for oh in 0..out_size {
                for ow in 0..out_size {
                    for ch in 0..self.c {
                        let g = got.get_f32(&[b, oh, ow, ch]);
                        let w = want[((b * out_size + oh) * out_size + ow) * self.c + ch];
                        if g != w {
                            return Err(format!("({b},{oh},{ow},{ch}) got {g} want {w}"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Under every tiling the benchmark catalogue offers, plus the default. A tiling only the
    /// sweep reaches is still a tiling that can be selected, so it is still one that has to be
    /// right.
    fn check_every_tiling(&self) {
        let mut bad = Vec::new();
        for rows in [1, 2, 4] {
            for cols in [1, 3, 4] {
                for (chans, lines) in [(1, 1), (2, 1), (3, 1), (1, 2), (1, 4), (2, 4)] {
                    let t = DepthwiseTiling {
                        rows,
                        cols,
                        chans,
                        lines,
                    };
                    if let Err(e) = self.check(t) {
                        bad.push(format!("{t:?}: {e}"));
                    }
                }
            }
        }
        if let Err(e) = self.check(DepthwiseTiling::default()) {
            bad.push(format!("default: {e}"));
        }
        assert!(bad.is_empty(), "failing tilings:\n{}", bad.join("\n"));
    }
}

/// The shape a depthwise layer actually has: a 3x3 window, unit stride, padded to keep the map.
#[test]
fn depthwise_3x3_padded() {
    Case {
        b: 2,
        c: 40,
        size: 7,
        k: 3,
        stride: 1,
        dilation: 1,
        padding: 1,
    }
    .check_every_tiling();
}

/// Unpadded, which takes the `checked` guard off every read.
#[test]
fn depthwise_3x3_unpadded() {
    Case {
        b: 1,
        c: 33,
        size: 6,
        k: 3,
        stride: 1,
        dilation: 1,
        padding: 0,
    }
    .check_every_tiling();
}

/// Strided, which is how the encoder's early stages downsample.
#[test]
fn depthwise_3x3_stride_2() {
    Case {
        b: 2,
        c: 32,
        size: 9,
        k: 3,
        stride: 2,
        dilation: 1,
        padding: 1,
    }
    .check_every_tiling();
}

/// A dilated 5x5: the shape seven of B4's blocks run, and the widest halo here.
#[test]
fn depthwise_5x5_dilated() {
    Case {
        b: 1,
        c: 64,
        size: 8,
        k: 5,
        stride: 1,
        dilation: 2,
        padding: 4,
    }
    .check_every_tiling();
}

/// A channel count that is not a multiple of the lane count, so the last cube's channel tile is
/// short and the walk has to stop where the tensor does.
#[test]
fn depthwise_channels_not_a_whole_number_of_cubes() {
    Case {
        b: 1,
        c: 37,
        size: 5,
        k: 3,
        stride: 1,
        dilation: 1,
        padding: 1,
    }
    .check_every_tiling();
}

/// Re-laying the filter must follow the binding's actual address map, not assume contiguous
/// `[C, kh, kw, 1]` storage.
#[test]
fn depthwise_strided_weight() {
    Case {
        b: 1,
        c: 8,
        size: 5,
        k: 3,
        stride: 1,
        dilation: 1,
        padding: 1,
    }
    .check_with_weight_gap(DepthwiseTiling::default(), 2)
    .unwrap();
}

/// The filter must carry exactly one channel's filter for every input/output channel. Reject the
/// mismatch before re-laying the binding under a larger logical shape.
#[test]
fn depthwise_rejects_mismatched_weight_shape() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let input_channels = 8;

    let input = TestInput::builder(client.clone(), Shape::new([1, 5, 5, input_channels]))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), Shape::new([1, 5, 5, input_channels]))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    for (weight_shape, want_weight_channels, want_group_channels) in
        [([7, 3, 3, 1], 7, 1), ([8, 3, 3, 2], 8, 2)]
    {
        let weight = TestInput::builder(client.clone(), Shape::new(weight_shape))
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();
        let result = launch_depthwise::<TestRuntime>(
            &client,
            DepthwiseTensors {
                input: input.clone().binding(),
                weight: weight.binding(),
                out: out.clone().binding(),
            },
            ConvolutionArgs::<2> {
                stride: [1; 2],
                padding: [1; 2],
                dilation: [1; 2],
            },
            input_channels,
            dtype,
            DepthwiseStrategy::Routine,
        );

        assert!(matches!(
            result,
            Err(ConvSetupError::NotDepthwise {
                groups: 8,
                input_channels: 8,
                output_channels: 8,
                weight_channels,
                weight_group_channels,
            }) if weight_channels == want_weight_channels
                && weight_group_channels == want_group_channels
        ));
    }
}
