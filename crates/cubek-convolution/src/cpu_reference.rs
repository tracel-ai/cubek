//! CPU reference and seeded "produce a HostData" primitives for 2D convolution.
//!
//! Inputs are laid out NHWC (input) and OHWI (weight) — the same convention the
//! test suite uses. Both `strategy_result` and `cpu_reference_result` build the
//! same input bits from `(seed_lhs, seed_rhs)`, so their outputs are directly
//! comparable.

use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::AddressType,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides, shape},
};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput, ValidationResult,
    assert_equals_approx, launch_and_capture_outcome,
};

use crate::{
    ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    launch_ref,
};

/// Run `strategy` against the conv problem with seeded inputs and return its
/// output as a [`HostData`].
pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    spec: ConvSpec,
    strategy: Strategy,
    dtypes: MatmulElems,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let inputs = seed_inputs(&client, &spec, &dtypes, seed_lhs, seed_rhs);
    let out_handle = inputs.out.clone();

    let outcome = launch_and_capture_outcome(&client, |c| {
        let conv_inputs = ConvolutionInputs::Forward {
            input: InputBinding::new(inputs.input.clone().binding(), dtypes.lhs_global),
            weight: InputBinding::new(inputs.weight.clone().binding(), dtypes.rhs_global),
            bias: None,
            out: out_handle.clone().binding(),
        };
        match launch_ref::<TestRuntime, 2>(
            &strategy,
            c,
            conv_inputs,
            spec.args.clone(),
            dtypes.clone(),
        ) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(e) => ExecutionOutcome::CompileError(format!("{e:?}")),
        }
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            out_handle,
            HostDataType::F32,
        )),
    }
}

/// CPU-only counterpart to [`strategy_result`]: generate the same seeded
/// inputs, run the naive CPU convolution, return the result as a [`HostData`].
pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    spec: ConvSpec,
    dtypes: MatmulElems,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let inputs = seed_inputs(&client, &spec, &dtypes, seed_lhs, seed_rhs);
    let problem = build_problem(&spec, &dtypes, &inputs);
    Ok(conv_cpu_reference(
        &inputs.input_data,
        &inputs.weight_data,
        &problem,
    ))
}

/// All the parameters needed to specify a 2D forward convolution problem,
/// independent of the seed/input data.
#[derive(Clone, Debug)]
pub struct ConvSpec {
    pub batches: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub out_channels: usize,
    pub args: ConvolutionArgs<2>,
    pub kernel_size: [usize; 2],
}

impl ConvSpec {
    pub fn out_h(&self) -> usize {
        let [s_h, _] = self.args.stride;
        let [p_h, _] = self.args.padding;
        let [d_h, _] = self.args.dilation;
        let k_h = self.kernel_size[0];
        (self.in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1
    }

    pub fn out_w(&self) -> usize {
        let [_, s_w] = self.args.stride;
        let [_, p_w] = self.args.padding;
        let [_, d_w] = self.args.dilation;
        let k_w = self.kernel_size[1];
        (self.in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1
    }
}

struct SeededInputs {
    input: TensorHandle<TestRuntime>,
    input_data: HostData,
    weight: TensorHandle<TestRuntime>,
    weight_data: HostData,
    out: TensorHandle<TestRuntime>,
}

fn seed_inputs(
    client: &ComputeClient<TestRuntime>,
    spec: &ConvSpec,
    dtypes: &MatmulElems,
    seed_lhs: u64,
    seed_rhs: u64,
) -> SeededInputs {
    let input_shape: Shape = shape![spec.batches, spec.in_h, spec.in_w, spec.channels];
    let weight_shape: Shape = shape![
        spec.out_channels,
        spec.kernel_size[0],
        spec.kernel_size[1],
        spec.channels
    ];
    let out_shape: Shape = shape![spec.batches, spec.out_h(), spec.out_w(), spec.out_channels];

    let (input, input_data) = TestInput::builder(client.clone(), input_shape)
        .dtype(dtypes.lhs_global)
        .uniform(seed_lhs, -1., 1.)
        .generate_with_f32_host_data();

    let (weight, weight_data) = TestInput::builder(client.clone(), weight_shape)
        .dtype(dtypes.rhs_global)
        .uniform(seed_rhs, -1., 1.)
        .generate_with_f32_host_data();

    let out = TestInput::builder(client.clone(), out_shape)
        .dtype(dtypes.acc_global)
        .zeros()
        .generate_without_host_data();

    SeededInputs {
        input,
        input_data,
        weight,
        weight_data,
        out,
    }
}

fn build_problem(
    spec: &ConvSpec,
    dtypes: &MatmulElems,
    inputs: &SeededInputs,
) -> ConvolutionProblem {
    let kernel_size_u32: Vec<u32> = spec.kernel_size.iter().map(|&v| v as u32).collect();
    let stride_u32: Vec<u32> = spec.args.stride.iter().map(|&v| v as u32).collect();
    let padding_i32: Vec<i32> = spec.args.padding.iter().map(|&v| v as i32).collect();
    let dilation_u32: Vec<u32> = spec.args.dilation.iter().map(|&v| v as u32).collect();

    let m = spec.batches * spec.out_h() * spec.out_w();
    let k = spec.kernel_size[0] * spec.kernel_size[1] * spec.channels;
    let n = spec.out_channels;

    let lhs_layout = MatrixLayout::RowMajor;
    let rhs_layout = MatrixLayout::RowMajor;

    let lhs_strides = inputs.input.strides().clone();
    let rhs_strides = inputs.weight.strides().clone();
    let _: &Strides = &lhs_strides;

    ConvolutionProblem {
        m,
        n,
        k,
        lhs_strides,
        rhs_strides,
        lhs_layout,
        rhs_layout,
        kernel_size: kernel_size_u32,
        stride: stride_u32,
        padding: padding_i32,
        dilation: dilation_u32,
        batches: spec.batches,
        in_shape: shape![spec.in_h, spec.in_w],
        channels: spec.channels,
        out_channels: spec.out_channels,
        padded_channels: spec.channels,
        out_shape: shape![spec.out_h(), spec.out_w()],
        dimensionality: Dimensionality::Dim2,
        operation: ConvolutionOperation::Forward,
        global_dtypes: MatmulGlobalElems {
            lhs: dtypes.lhs_global,
            rhs: dtypes.rhs_global,
            out: dtypes.acc_global,
        },
        address_type: AddressType::U32,
    }
}

/// Mirror of [`assert_equals_approx`] specialized to convolution; same epsilon
/// rules as the existing test helper.
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
