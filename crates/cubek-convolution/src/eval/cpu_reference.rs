//! CPU reference and seeded "produce a HostData" primitives for 2D convolution.
//!
//! Inputs are laid out NHWC (input) and OHWI (weight): the same convention the
//! test suite uses.

use cubecl::{
    client::Client,
    ir::AddressType,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides, shape},
};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, Progress, StridedLayout, TestInput,
    ValidationResult, assert_equals_approx, launch_and_capture_outcome,
};

use crate::{
    ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    launch_ref,
};

const HOST_HAS_NO_TRANSPOSED: &str = "the host states no transposed forward";

/// Run `strategy` against `operation` over `spec`'s geometry with seeded operands and return its
/// output as a [`HostData`]: the forward's activation, the weight gradient or the data gradient.
pub fn strategy_result(
    client: Client,
    spec: ConvSpec,
    operation: ConvolutionOperation,
    strategy: Strategy,
    dtypes: MatmulElems,
    seeds: [u64; 2],
) -> Result<HostData, String> {
    let inputs = seed_inputs(&client, &spec, operation, &dtypes, seeds)?;
    let out_handle = inputs.out.clone();

    let outcome = launch_and_capture_outcome(&client, &[&out_handle.handle], |c| {
        let lhs = InputBinding::new(inputs.lhs.clone().binding(), dtypes.lhs_global);
        let rhs = InputBinding::new(inputs.rhs.clone().binding(), dtypes.rhs_global);
        let out = out_handle.clone().binding();
        use ConvolutionOperation as Operation;
        let conv_inputs = match operation {
            Operation::Forward | Operation::ForwardTransposed => ConvolutionInputs::Forward {
                input: lhs,
                weight: rhs,
                bias: None,
                out,
            },
            Operation::BackwardWeight => ConvolutionInputs::BackwardWeight {
                input: lhs,
                out_grad: rhs,
                weight_grad: out,
            },
            Operation::BackwardData => ConvolutionInputs::BackwardData {
                out_grad: lhs,
                weights: rhs,
                in_grad: out,
            },
        };
        match launch_ref::<2>(&strategy, c, conv_inputs, spec.args, dtypes.clone()) {
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
/// operands, run the naive CPU convolution of `operation`, return the result as a [`HostData`].
pub fn cpu_reference_result(
    client: Client,
    spec: ConvSpec,
    operation: ConvolutionOperation,
    dtypes: MatmulElems,
    seeds: [u64; 2],
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    let inputs = seed_inputs(&client, &spec, operation, &dtypes, seeds)?;
    let args = &spec.args;
    let lhs_host = host_tensor(&inputs.lhs_data)?;
    let rhs_host = host_tensor(&inputs.rhs_data)?;
    let in_extent = [spec.in_h, spec.in_w];
    use ConvolutionOperation as Operation;
    let values = match operation {
        Operation::Forward => return Ok(forward_reference(&spec, &dtypes, &inputs, progress)),
        Operation::BackwardWeight => naive_weight_grad(lhs_host, rhs_host, spec.kernel_size, args),
        Operation::BackwardData => naive_data_grad(lhs_host, rhs_host, in_extent, args),
        Operation::ForwardTransposed => return Err(HOST_HAS_NO_TRANSPOSED.to_string()),
    };
    let shape = spec.shapes(operation)[2].clone();
    let strides = StridedLayout::RowMajor.compute_strides(&shape);
    Ok(HostData {
        data: HostDataVec::F32(values),
        shape,
        strides,
    })
}

/// Number of output writes [`conv_cpu_reference`] will produce for `problem`.
/// Matches the value the function sets on its [`Progress`] handle.
pub fn conv_cpu_reference_total(problem: &ConvolutionProblem) -> u64 {
    (problem.batches * problem.out_shape[0] * problem.out_shape[1] * problem.n) as u64
}

/// All the parameters needed to specify a 2D convolution problem,
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
        self.out()[0]
    }

    pub fn out_w(&self) -> usize {
        self.out()[1]
    }

    fn out(&self) -> [usize; 2] {
        let [window_h, window_w] = self.args.windows();
        let [kh, kw] = self.kernel_size;
        [
            window_h.output(self.in_h, kh),
            window_w.output(self.in_w, kw),
        ]
    }

    /// The shapes of `operation`'s lhs, rhs and out, the weight and its gradient `[co, kh, kw, ci]`
    /// and the rest NHWC: the input, the weight and the output of a forward; the input, the output
    /// gradient and the weight gradient of a weight backward; the output gradient, the weight and
    /// the input gradient of a data backward.
    fn shapes(&self, operation: ConvolutionOperation) -> [Shape; 3] {
        let input = shape![self.batches, self.in_h, self.in_w, self.channels];
        let weight = shape![
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.channels
        ];
        let out_grad = shape![self.batches, self.out_h(), self.out_w(), self.out_channels];
        use ConvolutionOperation as Operation;
        match operation {
            Operation::Forward => [input, weight, out_grad],
            Operation::BackwardWeight => [input, out_grad, weight],
            Operation::BackwardData | Operation::ForwardTransposed => [out_grad, weight, input],
        }
    }

    /// Evaluates this convolution on the host, plus `bias`; the weight's input channels set the
    /// groups.
    pub fn cpu_reference(
        &self,
        input: &HostData,
        weight: &HostData,
        bias: Option<&HostData>,
    ) -> HostData {
        let progress = None;
        ConvGeometry::from(self).cpu_reference(input, weight, bias, progress)
    }
}

fn host_tensor(data: &HostData) -> Result<HostTensor<'_>, String> {
    match &data.data {
        HostDataVec::F32(values) => Ok(HostTensor {
            values,
            dims: &data.shape,
        }),
        _ => Err("the host reference reads f32 operands".to_string()),
    }
}

/// The operands of an operation by the role they play: the lhs, the rhs and the zeroed output.
struct SeededInputs {
    lhs: TensorHandle,
    lhs_data: HostData,
    rhs: TensorHandle,
    rhs_data: HostData,
    out: TensorHandle,
}

fn seed_inputs(
    client: &Client,
    spec: &ConvSpec,
    operation: ConvolutionOperation,
    dtypes: &MatmulElems,
    [seed_lhs, seed_rhs]: [u64; 2],
) -> Result<SeededInputs, String> {
    if operation == ConvolutionOperation::ForwardTransposed {
        return Err("the host states no transposed forward".to_string());
    }
    let [lhs_shape, rhs_shape, out_shape] = spec.shapes(operation);

    let (lhs, lhs_data) = TestInput::builder(client.clone(), lhs_shape)
        .dtype(dtypes.lhs_global)
        .uniform(seed_lhs, -1., 1.)
        .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::builder(client.clone(), rhs_shape)
        .dtype(dtypes.rhs_global)
        .uniform(seed_rhs, -1., 1.)
        .generate_with_f32_host_data();

    let out = TestInput::builder(client.clone(), out_shape)
        .dtype(dtypes.acc_global)
        .zeros()
        .generate_without_host_data();

    Ok(SeededInputs {
        lhs,
        lhs_data,
        rhs,
        rhs_data,
        out,
    })
}

fn forward_reference(
    spec: &ConvSpec,
    dtypes: &MatmulElems,
    inputs: &SeededInputs,
    progress: Option<&Progress>,
) -> HostData {
    let problem = build_problem(spec, dtypes, inputs);
    conv_cpu_reference(&inputs.lhs_data, &inputs.rhs_data, &problem, progress)
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

    let lhs_strides = inputs.lhs.strides().clone();
    let rhs_strides = inputs.rhs.strides().clone();
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
    client: &Client,
    out: TensorHandle,
    dtypes: MatmulElems,
) -> ValidationResult {
    let epsilon = conv_epsilon(&dtypes, 500.);
    let expected = conv_cpu_reference(lhs, rhs, problem, None);
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
    progress: Option<&Progress>,
) -> HostData {
    let bias = None;
    ConvGeometry::from(problem).cpu_reference(lhs, rhs, bias, progress)
}

/// The spatial geometry of a 2D convolution, all but the input size, which the input carries.
struct ConvGeometry {
    out: [usize; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    /// At the start of each spatial dimension: a problem may pad the end differently, which `out`
    /// already reflects.
    padding: [i32; 2],
    dilation: [usize; 2],
}

impl ConvGeometry {
    /// NHWC `input` by OHWI `weight`, grouped when the weight holds fewer input channels.
    fn cpu_reference(
        &self,
        input: &HostData,
        weight: &HostData,
        bias: Option<&HostData>,
        progress: Option<&Progress>,
    ) -> HostData {
        let [n, h, w, c] = [0, 1, 2, 3].map(|axis| input.shape[axis]);
        let (out_channels, c_in_group) = (weight.shape[0], weight.shape[3]);
        let c_out_group = out_channels / (c / c_in_group);
        let Self {
            out: [out_h, out_w],
            kernel: [kh, kw],
            stride,
            padding,
            dilation,
        } = *self;

        if let Some(p) = progress {
            p.set_total((n * out_h * out_w * out_channels) as u64);
        }

        let mut out = vec![0.0_f32; n * out_h * out_w * out_channels];

        for nth_batch in 0..n {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    for out_c in 0..out_channels {
                        let group_start = out_c / c_out_group * c_in_group;
                        let mut acc = bias.map_or(0.0, |bias| bias.get_f32(&[out_c]));
                        for in_c in 0..c_in_group {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let in_y = out_y as i32 * stride[0] as i32
                                        + ky as i32 * dilation[0] as i32
                                        - padding[0];
                                    let in_x = out_x as i32 * stride[1] as i32
                                        + kx as i32 * dilation[1] as i32
                                        - padding[1];

                                    if in_y >= 0 && in_y < h as i32 && in_x >= 0 && in_x < w as i32
                                    {
                                        let value = input.get_f32(&[
                                            nth_batch,
                                            in_y as usize,
                                            in_x as usize,
                                            group_start + in_c,
                                        ]);
                                        let weight = weight.get_f32(&[out_c, ky, kx, in_c]);

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
                        if let Some(p) = progress {
                            p.bump();
                        }
                    }
                }
            }
        }

        let out_shape: Shape = shape![n, out_h, out_w, out_channels];
        let strides = StridedLayout::RowMajor.compute_strides(&out_shape);

        HostData {
            data: HostDataVec::F32(out),
            shape: out_shape,
            strides,
        }
    }
}

impl From<&ConvolutionProblem> for ConvGeometry {
    fn from(problem: &ConvolutionProblem) -> Self {
        Self {
            out: [problem.out_shape[0], problem.out_shape[1]],
            kernel: [0, 1].map(|axis| problem.kernel_size[axis] as usize),
            stride: [0, 1].map(|axis| problem.stride[axis] as usize),
            padding: [problem.padding[0], problem.padding[1]],
            dilation: [0, 1].map(|axis| problem.dilation[axis] as usize),
        }
    }
}

impl From<&ConvSpec> for ConvGeometry {
    fn from(spec: &ConvSpec) -> Self {
        Self {
            out: [spec.out_h(), spec.out_w()],
            kernel: spec.kernel_size,
            stride: spec.args.stride,
            padding: spec.args.padding.map(|padding| padding as i32),
            dilation: spec.args.dilation,
        }
    }
}

/// A host tensor as the naive references read it: its values and its dims, channels last but for
/// a weight and its gradient, which are `[co, taps.., ci]`.
#[derive(Debug, Clone, Copy)]
pub struct HostTensor<'a> {
    pub values: &'a [f32],
    pub dims: &'a [usize],
}

/// The multi-index `flat` names in a row-major box of `extents`.
fn unravel(mut flat: usize, extents: &[usize]) -> Vec<usize> {
    let mut index = vec![0; extents.len()];
    for (at, &extent) in index.iter_mut().zip(extents).rev() {
        *at = flat % extent;
        flat /= extent;
    }
    index
}

/// The row-major offset of the cell a window reads, at the positions of each spatial dim, `None`
/// where one lands outside `extents`.
fn within(extents: &[usize], positions: impl Iterator<Item = isize>) -> Option<usize> {
    extents
        .iter()
        .zip(positions)
        .try_fold(0, |flat, (&extent, at)| {
            let at = usize::try_from(at).ok().filter(|&at| at < extent)?;
            Some(flat * extent + at)
        })
}

/// `out[b, o.., co] = sum over taps, ci of input[b, o * stride + tap * dilation - padding.., ci] *
/// weight[co, tap.., ci]`, the taps that land outside the input skipped.
pub fn naive_forward<const N: usize>(
    input: HostTensor,
    weight: HostTensor,
    args: &ConvolutionArgs<N>,
) -> Vec<f32> {
    let windows = args.windows();
    let (batches, channels) = (input.dims[0], input.dims[N + 1]);
    let (gathered, kernel) = (&input.dims[1..=N], &weight.dims[1..=N]);
    let filters = weight.dims[0];
    let out: Vec<usize> = (0..N)
        .map(|dim| windows[dim].output(gathered[dim], kernel[dim]))
        .collect();
    let (cells, taps, inputs) = (
        out.iter().product::<usize>(),
        kernel.iter().product::<usize>(),
        gathered.iter().product::<usize>(),
    );
    let mut result = vec![0.0; batches * cells * filters];
    for b in 0..batches {
        for cell in 0..cells {
            let at = unravel(cell, &out);
            for o in 0..filters {
                let mut sum = 0.0;
                for tap in 0..taps {
                    let at_tap = unravel(tap, kernel);
                    let positions = (0..N).map(|dim| windows[dim].position(at[dim], at_tap[dim]));
                    let Some(read) = within(gathered, positions) else {
                        continue;
                    };
                    for i in 0..channels {
                        sum += input.values[(b * inputs + read) * channels + i]
                            * weight.values[(o * taps + tap) * channels + i];
                    }
                }
                result[(b * cells + cell) * filters + o] = sum;
            }
        }
    }
    result
}

/// `weight_grad[co, tap.., ci] = sum over b, o.. of input[b, o * stride + tap * dilation -
/// padding.., ci] * out_grad[b, o.., co]`, the taps that land outside the input skipped.
pub fn naive_weight_grad<const N: usize>(
    input: HostTensor,
    out_grad: HostTensor,
    kernel: [usize; N],
    args: &ConvolutionArgs<N>,
) -> Vec<f32> {
    let windows = args.windows();
    let (batches, channels) = (input.dims[0], input.dims[N + 1]);
    let (gathered, out) = (&input.dims[1..=N], &out_grad.dims[1..=N]);
    let filters = out_grad.dims[N + 1];
    let (cells, taps, inputs) = (
        out.iter().product::<usize>(),
        kernel.iter().product::<usize>(),
        gathered.iter().product::<usize>(),
    );
    let mut grad = vec![0.0; filters * taps * channels];
    for o in 0..filters {
        for tap in 0..taps {
            let at_tap = unravel(tap, &kernel);
            for i in 0..channels {
                let mut sum = 0.0;
                for b in 0..batches {
                    for cell in 0..cells {
                        let at = unravel(cell, out);
                        let positions =
                            (0..N).map(|dim| windows[dim].position(at[dim], at_tap[dim]));
                        let Some(read) = within(gathered, positions) else {
                            continue;
                        };
                        sum += input.values[(b * inputs + read) * channels + i]
                            * out_grad.values[(b * cells + cell) * filters + o];
                    }
                }
                grad[(o * taps + tap) * channels + i] = sum;
            }
        }
    }
    grad
}

/// `in_grad[b, p.., ci] = sum over co, taps of out_grad[b, g.., co] * weights[co, tap.., ci]`
/// over the output positions `g` whose window reads input `p` at `tap`, that is
/// `g * stride = p + padding - tap * dilation`; the others are skipped.
pub fn naive_data_grad<const N: usize>(
    out_grad: HostTensor,
    weights: HostTensor,
    input: [usize; N],
    args: &ConvolutionArgs<N>,
) -> Vec<f32> {
    let (batches, filters) = (out_grad.dims[0], out_grad.dims[N + 1]);
    let (out, kernel) = (&out_grad.dims[1..=N], &weights.dims[1..=N]);
    let channels = weights.dims[N + 1];
    let (cells, taps, inputs) = (
        out.iter().product::<usize>(),
        kernel.iter().product::<usize>(),
        input.iter().product::<usize>(),
    );
    let mut grad = vec![0.0; batches * inputs * channels];
    for b in 0..batches {
        for position in 0..inputs {
            let at = unravel(position, &input);
            for i in 0..channels {
                let mut sum = 0.0;
                for tap in 0..taps {
                    let at_tap = unravel(tap, kernel);
                    let read = out.iter().enumerate().try_fold(0, |flat, (dim, &extent)| {
                        let at = output_reading(args, dim, at[dim], at_tap[dim], extent)?;
                        Some(flat * extent + at)
                    });
                    let Some(read) = read else {
                        continue;
                    };
                    for o in 0..filters {
                        sum += out_grad.values[(b * cells + read) * filters + o]
                            * weights.values[(o * taps + tap) * channels + i];
                    }
                }
                grad[(b * inputs + position) * channels + i] = sum;
            }
        }
    }
    grad
}

/// The output position along spatial `dim` whose window, at tap `tap`, reads input position `pos`
/// of an output `extent` long, if there is one.
fn output_reading<const N: usize>(
    args: &ConvolutionArgs<N>,
    dim: usize,
    pos: usize,
    tap: usize,
    extent: usize,
) -> Option<usize> {
    let reach = (pos + args.padding[dim]) as isize - (tap * args.dilation[dim]) as isize;
    let stride = args.stride[dim] as isize;
    let at = usize::try_from(reach)
        .ok()
        .filter(|_| reach % stride == 0)?
        / stride as usize;
    (at < extent).then_some(at)
}
