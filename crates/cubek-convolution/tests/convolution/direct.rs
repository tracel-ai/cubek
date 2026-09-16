//! The direct routine against a CPU reference.
//!
//! Most layers here have more input channels per group than a vector has lanes, on purpose: only
//! those take the CPU path that blocks output channels, and a grid of narrow layers would pass
//! however that path summed.

use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// One convolution, in the layout the routine takes: NHWC in and out, `[C_out, kh, kw, C_in /
/// groups]` weights.
#[derive(Debug, Clone, Copy)]
struct Case {
    b: usize,
    c_in: usize,
    c_out: usize,
    groups: usize,
    /// The side of the input map.
    size: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    dilation: usize,
    /// Padding at the beginning and end of each spatial dimension.
    padding: usize,
    bias: bool,
}

impl Case {
    fn out_size(&self, k: usize) -> usize {
        (self.size + 2 * self.padding - self.dilation * (k - 1) - 1) / self.stride + 1
    }

    /// Small integers on a short period. Every product is at most 2 and every partial sum stays
    /// under 2048, so the accumulation is exact even in `f16` and any summation order is equal.
    fn ramp(n: usize, period: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
    }

    fn reference(&self, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let (oh_n, ow_n) = (self.out_size(self.kh), self.out_size(self.kw));
        let c_in_group = self.c_in / self.groups;
        let c_out_group = self.c_out / self.groups;
        let mut result = vec![0.0f32; self.b * oh_n * ow_n * self.c_out];

        for b in 0..self.b {
            for oh in 0..oh_n {
                for ow in 0..ow_n {
                    for oc in 0..self.c_out {
                        let group = oc / c_out_group;
                        let mut acc = if self.bias { bias[oc] } else { 0.0 };
                        for ic in 0..c_in_group {
                            for rh in 0..self.kh {
                                for rw in 0..self.kw {
                                    let h = (oh * self.stride + rh * self.dilation) as isize
                                        - self.padding as isize;
                                    let w = (ow * self.stride + rw * self.dilation) as isize
                                        - self.padding as isize;
                                    if h < 0
                                        || w < 0
                                        || h >= self.size as isize
                                        || w >= self.size as isize
                                    {
                                        continue;
                                    }
                                    let x = input[((b * self.size + h as usize) * self.size
                                        + w as usize)
                                        * self.c_in
                                        + group * c_in_group
                                        + ic];
                                    let f = weight
                                        [((oc * self.kh + rh) * self.kw + rw) * c_in_group + ic];
                                    acc += x * f;
                                }
                            }
                        }
                        result[((b * oh_n + oh) * ow_n + ow) * self.c_out + oc] = acc;
                    }
                }
            }
        }

        result
    }

    fn check(&self, dtype: ElemType) -> Result<(), String> {
        let client = cubecl::test_device().client();
        let (oh_n, ow_n) = (self.out_size(self.kh), self.out_size(self.kw));

        let in_shape = [self.b, self.size, self.size, self.c_in];
        let w_shape = [self.c_out, self.kh, self.kw, self.c_in / self.groups];
        let out_shape = [self.b, oh_n, ow_n, self.c_out];

        let in_data = Self::ramp(in_shape.iter().product(), 4);
        let w_data = Self::ramp(w_shape.iter().product(), 3);
        let b_data = Self::ramp(self.c_out, 5);

        let (input, _) = TestInput::builder(client.clone(), Shape::new(in_shape))
            .dtype(dtype)
            .custom(in_data.clone())
            .generate_with_f32_host_data();
        let (weight, _) = TestInput::builder(client.clone(), Shape::new(w_shape))
            .dtype(dtype)
            .custom(w_data.clone())
            .generate_with_f32_host_data();
        let bias: Option<TensorHandle> = self.bias.then(|| {
            TestInput::builder(client.clone(), Shape::new([self.c_out]))
                .dtype(dtype)
                .custom(b_data.clone())
                .generate_with_f32_host_data()
                .0
        });
        // Zeroed, so a cell the kernel never writes reads as the zero it started as.
        let out: TensorHandle = TestInput::builder(client.clone(), Shape::new(out_shape))
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();

        launch_direct::<2>(
            &client,
            DirectTensors {
                input: input.binding(),
                weight: weight.binding(),
                bias: bias.map(|bias| bias.binding()),
                out: out.clone().binding(),
            },
            ConvolutionArgs::<2> {
                stride: [self.stride; 2],
                padding: [self.padding; 2],
                dilation: [self.dilation; 2],
            },
            self.groups,
            dtype,
        )
        .map_err(|e| format!("setup: {e:?}"))?;

        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let want = self.reference(&in_data, &w_data, &b_data);

        for b in 0..self.b {
            for oh in 0..oh_n {
                for ow in 0..ow_n {
                    for oc in 0..self.c_out {
                        let g = got.get_f32(&[b, oh, ow, oc]);
                        let w = want[((b * oh_n + oh) * ow_n + ow) * self.c_out + oc];
                        if g != w {
                            return Err(format!("({b},{oh},{ow},{oc}) got {g} want {w}"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// `(c_in, c_out, groups)`. The output counts reach output vectors of 16, 4, 2 and 1, so a
/// channel block that does not divide its vector, or drops a remainder, has a case to fail.
const CHANNELS: [(usize, usize, usize); 11] = [
    (3, 8, 1),
    (16, 16, 1),
    (32, 32, 1),
    (64, 64, 1),
    (64, 12, 1),
    (64, 6, 1),
    (64, 5, 1),
    (128, 32, 2),
    (32, 64, 2),
    (192, 48, 3),
    (104, 26, 1),
];

/// `(kh, kw, stride, dilation, padding)`.
const GEOMETRIES: [(usize, usize, usize, usize, usize); 6] = [
    (3, 3, 1, 1, 1),
    (3, 3, 2, 1, 1),
    (3, 3, 1, 2, 2),
    (3, 3, 1, 1, 0),
    (1, 1, 1, 1, 0),
    (1, 3, 1, 1, 1),
];

fn check_grid(dtype: ElemType) {
    let mut bad = Vec::new();

    for (c_in, c_out, groups) in CHANNELS {
        for (kh, kw, stride, dilation, padding) in GEOMETRIES {
            for bias in [false, true] {
                let case = Case {
                    b: 2,
                    c_in,
                    c_out,
                    groups,
                    size: 6,
                    kh,
                    kw,
                    stride,
                    dilation,
                    padding,
                    bias,
                };
                if let Err(e) = case.check(dtype) {
                    bad.push(format!("{case:?}: {e}"));
                }
            }
        }
    }

    assert!(bad.is_empty(), "failing cases:\n{}", bad.join("\n"));
}

#[test]
fn direct_f32_matches_the_reference() {
    check_grid(f32::elem_type_native());
}

/// An `f16` vector has twice the lanes of an `f32` one at the same load width, so the same layer
/// compiles a different kernel and needs twice the input channels to take the blocked path.
#[test]
fn direct_f16_matches_the_reference() {
    check_grid(half::f16::elem_type_native());
}
