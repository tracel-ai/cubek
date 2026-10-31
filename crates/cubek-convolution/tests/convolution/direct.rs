//! The direct routine against a CPU reference.
//!
//! Most layers here have more input channels per group than a vector has lanes, on purpose: only
//! those take the CPU path that blocks output channels, and a grid of narrow layers would pass
//! however that path summed.

use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_convolution::{
    ConvolutionArgs, DirectTensors, eval::cpu_reference::ConvSpec, launch_direct,
};
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
    /// Small integers on a short period. Every product is at most 2 and every partial sum stays
    /// under 2048, so the accumulation is exact even in `f16` and any summation order is equal.
    fn ramp(n: usize, period: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
    }

    /// Values on a period of 17 in `[-0.5, 0.5)`: their products round, so the sum's own
    /// precision shows.
    fn fractions(n: usize, period: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 + period) % 17) as f32 / 17.0 - 0.5)
            .collect()
    }

    /// Runs the case on `values` and compares it to the CPU reference, to within `tolerance`
    /// times one plus the reference's magnitude.
    fn check(
        &self,
        dtype: ElemType,
        values: fn(usize, usize) -> Vec<f32>,
        tolerance: f32,
    ) -> Result<(), String> {
        let client = cubecl::test_device().client();
        let args = ConvolutionArgs::<2> {
            stride: [self.stride; 2],
            padding: [self.padding; 2],
            dilation: [self.dilation; 2],
        };
        let spec = ConvSpec {
            batches: self.b,
            in_h: self.size,
            in_w: self.size,
            channels: self.c_in,
            out_channels: self.c_out,
            args: args.clone(),
            kernel_size: [self.kh, self.kw],
        };

        let in_shape = [self.b, self.size, self.size, self.c_in];
        let w_shape = [self.c_out, self.kh, self.kw, self.c_in / self.groups];
        let out_shape = [self.b, spec.out_h(), spec.out_w(), self.c_out];

        let (input, input_host) = TestInput::builder(client.clone(), Shape::new(in_shape))
            .dtype(dtype)
            .custom(values(in_shape.iter().product(), 4))
            .generate_with_f32_host_data();
        let (weight, weight_host) = TestInput::builder(client.clone(), Shape::new(w_shape))
            .dtype(dtype)
            .custom(values(w_shape.iter().product(), 3))
            .generate_with_f32_host_data();
        let (bias, bias_host) = self
            .bias
            .then(|| {
                TestInput::builder(client.clone(), Shape::new([self.c_out]))
                    .dtype(dtype)
                    .custom(values(self.c_out, 5))
                    .generate_with_f32_host_data()
            })
            .unzip();
        // No output reaches 4096 and f16 holds it exactly, so a cell the kernel never writes fails.
        let out: TensorHandle = TestInput::builder(client.clone(), Shape::new(out_shape))
            .dtype(dtype)
            .custom(vec![4096.0; out_shape.iter().product::<usize>()])
            .generate_without_host_data();

        launch_direct::<2>(
            &client,
            DirectTensors {
                input: input.binding(),
                weight: weight.binding(),
                bias: bias.map(|bias| bias.binding()),
                out: out.clone().binding(),
            },
            args,
            self.groups,
            dtype,
        )
        .map_err(|e| format!("setup: {e:?}"))?;

        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let want = spec.cpu_reference(&input_host, &weight_host, bias_host.as_ref());

        for b in 0..self.b {
            for oh in 0..out_shape[1] {
                for ow in 0..out_shape[2] {
                    for oc in 0..self.c_out {
                        let (g, w) = (
                            got.get_f32(&[b, oh, ow, oc]),
                            want.get_f32(&[b, oh, ow, oc]),
                        );
                        if g.is_nan() || (g - w).abs() > tolerance * (1.0 + w.abs()) {
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
                if let Err(e) = case.check(dtype, Case::ramp, 0.0) {
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

/// A half-precision layer of 320 input channels summed over a 3x3 window: 2880 products a sum,
/// which drift by a rounding a term when summed in `f16`. The reference sums the same `f16`
/// values in `f32`, so what is left is the output's own rounding, half a unit in its last place.
#[test]
fn direct_f16_sums_in_f32() {
    let case = Case {
        b: 1,
        c_in: 320,
        c_out: 8,
        groups: 1,
        size: 6,
        kh: 3,
        kw: 3,
        stride: 1,
        dilation: 1,
        padding: 1,
        bias: true,
    };
    case.check(half::f16::elem_type_native(), Case::fractions, 1e-3)
        .unwrap();
}
