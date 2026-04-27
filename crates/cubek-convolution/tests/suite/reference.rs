//! Naive CPU convolution reference

use cubecl::{
    TestRuntime,
    client::ComputeClient,
    std::tensor::TensorHandle,
    zspace::{Shape, shape},
};
use cubek_convolution::components::ConvolutionProblem;
use cubek_matmul::definition::MatmulElems;
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StrideSpec, ValidationResult, assert_equals_approx,
};

pub fn assert_result(
    lhs: &HostData,
    rhs: &HostData,
    problem: &ConvolutionProblem,
    client: &ComputeClient<TestRuntime>,
    out: TensorHandle<TestRuntime>,
    dtypes: MatmulElems,
) -> ValidationResult {
    let epsilon = conv_epsilon(&dtypes, 500.);
    let expected = conv_cpu_reference(lhs, rhs, problem);
    let actual = HostData::from_tensor_handle(client, out, HostDataType::F32);

    assert_equals_approx(&actual, &expected, epsilon)
}

fn conv_epsilon(elems: &MatmulElems, safety_factor: f32) -> f32 {
    let total_eps = elems
        .lhs_global
        .epsilon()
        .max(elems.rhs_global.epsilon())
        .max(elems.acc_global.epsilon())
        .max(elems.lhs_stage.epsilon())
        .max(elems.rhs_stage.epsilon())
        .max(elems.acc_stage.epsilon())
        .max(elems.lhs_register.epsilon())
        .max(elems.rhs_register.epsilon())
        .max(elems.acc_register.epsilon());

    total_eps as f32 * safety_factor
}

/// Naive CPU convolution: very slow on large payloads, only for testing.
///
/// All math is done in f32 against the host tensors carried in `HostData`.
pub fn conv_cpu_reference(
    lhs: &HostData,
    rhs: &HostData,
    problem: &ConvolutionProblem,
) -> HostData {
    let n = problem.batches;
    let h = problem.in_shape[0];
    let w = problem.in_shape[1];
    let c = problem.channels;

    let out_h = problem.out_shape[0];
    let out_w = problem.out_shape[1];
    let out_channels = problem.n;

    let kh = problem.kernel_size[0] as usize;
    let kw = problem.kernel_size[1] as usize;

    let padding = &problem.padding;
    let stride = &problem.stride;
    let dilation = &problem.dilation;

    let mut out = vec![0.0_f32; n * out_h * out_w * out_channels];

    for nth_batch in 0..n {
        for out_y in 0..out_h {
            for out_x in 0..out_w {
                for out_c in 0..out_channels {
                    let mut acc = 0.0_f32;
                    for in_c in 0..c {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let in_y = out_y as i32 * stride[0] as i32
                                    + ky as i32 * dilation[0] as i32
                                    - padding[0];
                                let in_x = out_x as i32 * stride[1] as i32
                                    + kx as i32 * dilation[1] as i32
                                    - padding[1];

                                if in_y >= 0 && in_y < h as i32 && in_x >= 0 && in_x < w as i32 {
                                    let value = lhs.get_f32(&[
                                        nth_batch,
                                        in_y as usize,
                                        in_x as usize,
                                        in_c,
                                    ]);
                                    let weight = rhs.get_f32(&[out_c, ky, kx, in_c]);

                                    acc += value * weight;
                                }
                            }
                        }
                    }
                    let out_linear = nth_batch * out_h * out_w * out_channels
                        + out_y * out_w * out_channels
                        + out_x * out_channels
                        + out_c;
                    out[out_linear] = acc;
                }
            }
        }
    }

    let out_shape: Shape = shape![n, out_h, out_w, out_channels];
    let strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    HostData {
        data: HostDataVec::F32(out),
        shape: out_shape,
        strides,
    }
}
