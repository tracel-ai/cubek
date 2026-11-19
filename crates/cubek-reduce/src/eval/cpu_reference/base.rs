use super::{
    reference_all, reference_any, reference_argmax, reference_argmin, reference_argtopk,
    reference_max, reference_max_abs, reference_mean, reference_min, reference_prod, reference_sum,
    reference_topk,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::{
    client::Client,
    prelude::*,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, Progress, TestInput,
    launch_and_capture_outcome,
};

use crate::ReduceOperation;
use crate::kernels::{self, ReduceElements, ReduceOpStrategy};

/// The bits a comparison runs on: the kernel launch and the CPU reference each
/// build their input from this alone, so the two see the same values.
#[derive(Clone, Copy)]
pub struct ReduceInput {
    pub dtype: ElemType,
    pub values: ReduceValues,
}

/// Where a comparison's input values come from.
#[derive(Clone, Copy)]
pub enum ReduceValues {
    /// Uniform over `[-1, 1]`, drawn from this seed.
    Uniform(u64),
    /// An evenly spaced ramp over `[-1, 1)`, distributed by an odd stride so no
    /// run of the tensor is sorted. Every value stays distinct down to f16, so
    /// a top-k has a single right answer and no tie can decide the comparison.
    Ramp,
}

/// The range a uniform input is drawn from.
const UNIFORM_LOW: f32 = -1.0;
const UNIFORM_HIGH: f32 = 1.0;

/// Ramp steps f16 still resolves apart: the ramp spans 2.0 and f16 separates
/// neighbours by 2^-11 just below 1.0.
pub const RAMP_MAX_ELEMS: usize = 2048;

/// Odd, so it visits every slot of a power-of-two length.
const RAMP_STRIDE_ELEMS: usize = 1103;

impl ReduceInput {
    /// Uniform values at `dtype`, the input every seeded comparison over a
    /// catalogue-sized shape uses.
    pub fn uniform(dtype: ElemType, seed: u64) -> Self {
        Self {
            dtype,
            values: ReduceValues::Uniform(seed),
        }
    }

    fn tensor(&self, client: Client, shape: Vec<usize>) -> TestInput {
        let builder = TestInput::builder(client, shape.clone()).dtype(self.dtype);
        match self.values {
            ReduceValues::Uniform(seed) => builder.uniform(seed, UNIFORM_LOW, UNIFORM_HIGH),
            ReduceValues::Ramp => builder.custom(ramp_of_len(shape.iter().product())),
        }
    }
}

/// A ramp of `elems` values over `[-1, 1)`, spread by [`RAMP_STRIDE_ELEMS`] so no run of it is
/// sorted.
fn ramp_of_len(elems: usize) -> Vec<f32> {
    assert!(
        elems.is_power_of_two() && elems <= RAMP_MAX_ELEMS,
        "a ramp of {elems} values is not a permutation, or is finer than f16 resolves"
    );
    (0..elems)
        .map(|i| (i * RAMP_STRIDE_ELEMS % elems) as f32 / elems as f32 * 2.0 - 1.0)
        .collect()
}

/// How far a kernel's output may sit from the reference before it is wrong.
///
/// A selection instruction returns a value taken out of the input, or its
/// coordinate, so any slack at all leaves the comparison unable to fail. Only
/// the accumulating ones round, and over tens of millions of f32 elements some
/// kernels accumulate noticeable noise; tightening those belongs in the
/// per-routine integration tests.
pub fn comparison_epsilon(operation: ReduceOperation) -> f32 {
    match operation {
        ReduceOperation::Sum | ReduceOperation::Mean | ReduceOperation::Prod => 1.0,
        ReduceOperation::Max
        | ReduceOperation::Min
        | ReduceOperation::MaxAbs
        | ReduceOperation::ArgMax
        | ReduceOperation::ArgMin
        | ReduceOperation::TopK { .. }
        | ReduceOperation::ArgTopK { .. }
        | ReduceOperation::Any
        | ReduceOperation::All => 0.0,
    }
}

/// Run `strategy` on `input` and return its output as a [`HostData`].
///
/// # Panics
///
/// Where `input` is a ramp whose shape holds a count of values that is not a power of two at
/// most [`RAMP_MAX_ELEMS`].
pub fn strategy_result(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    strategy: ReduceOpStrategy,
    operation: ReduceOperation,
    input: ReduceInput,
) -> Result<HostData, String> {
    let input_dtype = input.dtype;
    let output_dtype = output_dtype_for(operation, input_dtype);
    let accumulation_dtype = accumulation_dtype();

    let input_handle = input
        .tensor(client.clone(), shape.clone())
        .generate_without_host_data();

    let out_shape = output_shape_for(&shape, axis, operation);
    let output_handle = zeroed(&client, out_shape, output_dtype);

    let elements = ReduceElements {
        input: input_dtype,
        accumulator: accumulation_dtype,
        out: output_dtype,
    };

    let outcome = launch_and_capture_outcome(&client, &[&output_handle.handle], |c| {
        kernels::Reduce::new(strategy)
            .execute(
                c,
                &cubecl::test_device(),
                input_handle.clone().binding(),
                output_handle.clone().binding(),
                axis,
                operation,
                elements,
            )
            .into()
    });

    read_back(&client, outcome, output_handle)
}

/// Like [`strategy_result`], but runs the fused `Reduce::execute_with_indices` path and
/// returns its **values** output.
///
/// The benchmark catalogue times the fused path, so it has to be able to
/// validate the fused path too: running the plain `Reduce::execute` here instead would
/// silently check a kernel that is not the one being measured.
///
/// # Panics
///
/// As [`strategy_result`].
pub fn strategy_result_with_indices(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    strategy: ReduceOpStrategy,
    operation: ReduceOperation,
    input: ReduceInput,
) -> Result<HostData, String> {
    let input_dtype = input.dtype;
    let index_dtype = u32::elem_type_native();
    let accumulation_dtype = accumulation_dtype();

    let input_handle = input
        .tensor(client.clone(), shape.clone())
        .generate_without_host_data();

    let out_shape = output_shape_for(&shape, axis, operation);
    let values_handle = zeroed(&client, out_shape.clone(), input_dtype);
    let indices_handle = zeroed(&client, out_shape, index_dtype);

    let elements = ReduceElements {
        input: input_dtype,
        accumulator: accumulation_dtype,
        out: input_dtype,
    };

    let outcome = launch_and_capture_outcome(
        &client,
        &[&values_handle.handle, &indices_handle.handle],
        |c| {
            kernels::Reduce::new(strategy)
                .execute_with_indices(
                    c,
                    &cubecl::test_device(),
                    input_handle.clone().binding(),
                    values_handle.clone().binding(),
                    indices_handle.clone().binding(),
                    axis,
                    operation,
                    elements,
                    index_dtype,
                )
                .into()
        },
    );

    read_back(&client, outcome, values_handle)
}

/// A buffer of `shape` in `dtype`, holding zeros.
fn zeroed(client: &Client, shape: Vec<usize>, dtype: ElemType) -> TensorHandle {
    TestInput::builder(client.clone(), shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

/// `handle` read back as f32 where the launch ran, or why it did not compile.
fn read_back(
    client: &Client,
    outcome: ExecutionOutcome,
    handle: TensorHandle,
) -> Result<HostData, String> {
    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            client,
            handle,
            HostDataType::F32,
        )),
    }
}

/// CPU-only counterpart to [`strategy_result`]: generate the same seeded
/// inputs, run the matching naive reduce reference, return the result as a
/// [`HostData`].
///
/// # Panics
///
/// As [`strategy_result`].
pub fn cpu_reference_result(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    operation: ReduceOperation,
    input: ReduceInput,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    if let Some(progress) = progress {
        progress.set_total(cpu_reference_total(&shape, axis, operation));
    }

    // A narrow dtype rounds on the way in, so the reference folds what the
    // tensor ended up holding rather than what the generator was handed.
    let (_input_handle, input_host) = input.tensor(client, shape).generate_with_f32_host_data();

    Ok(reference_for_operation(
        &input_host,
        axis,
        operation,
        progress,
    ))
}

fn reference_for_operation(
    input: &HostData,
    axis: usize,
    operation: ReduceOperation,
    progress: Option<&Progress>,
) -> HostData {
    match operation {
        ReduceOperation::Sum => reference_sum(input, axis, progress),
        ReduceOperation::Mean => reference_mean(input, axis, progress),
        ReduceOperation::Prod => reference_prod(input, axis, progress),
        ReduceOperation::Min => reference_min(input, axis, progress),
        ReduceOperation::Max => reference_max(input, axis, progress),
        ReduceOperation::MaxAbs => reference_max_abs(input, axis, progress),
        ReduceOperation::ArgMax => reference_argmax(input, axis, progress),
        ReduceOperation::ArgMin => reference_argmin(input, axis, progress),
        ReduceOperation::ArgTopK { k } => reference_argtopk(input, axis, k, progress),
        ReduceOperation::TopK { k } => reference_topk(input, axis, k, progress),
        ReduceOperation::Any => reference_any(input, axis, progress),
        ReduceOperation::All => reference_all(input, axis, progress),
    }
}

/// Number of progress bumps the reduce reference will produce: one per output
/// cell.
pub fn cpu_reference_total(shape: &[usize], axis: usize, operation: ReduceOperation) -> u64 {
    let out_shape = output_shape_for(shape, axis, operation);
    out_shape.iter().product::<usize>() as u64
}

fn output_shape_for(shape: &[usize], axis: usize, operation: ReduceOperation) -> Vec<usize> {
    let mut out = shape.to_vec();
    out[axis] = operation.written_per_row();
    out
}

/// What a `Reduce::execute` of `operation` writes: a coordinate for the `Arg*` family, the
/// input's own element type for everything else.
pub fn output_dtype_for(operation: ReduceOperation, input_dtype: ElemType) -> ElemType {
    match operation {
        ReduceOperation::ArgMax | ReduceOperation::ArgMin | ReduceOperation::ArgTopK { .. } => {
            u32::elem_type_native()
        }
        _ => input_dtype,
    }
}

/// The accumulator element the harness states for every launch it compares, whatever the input or
/// output element.
pub fn accumulation_dtype() -> ElemType {
    f32::elem_type_native()
}

pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let n = shape.len();
    if n == 0 {
        return Strides::new(&[] as &[usize]);
    }
    let mut s = vec![0usize; n];
    s[n - 1] = 1;
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    Strides::new(&s)
}

pub(crate) fn output_shape(input_shape: &Shape, axis: usize) -> Vec<usize> {
    let mut out: Vec<usize> = input_shape.iter().copied().collect();
    out[axis] = 1;
    out
}

pub(super) fn should_replace_max(current: f32, candidate: f32) -> bool {
    !current.is_nan() && (candidate.is_nan() || candidate > current)
}

pub(super) fn should_replace_min(current: f32, candidate: f32) -> bool {
    !current.is_nan() && (candidate.is_nan() || candidate < current)
}

pub(crate) fn for_each_output_coord(output_shape: &[usize], mut f: impl FnMut(usize, &[usize])) {
    let rank = output_shape.len();
    if rank == 0 {
        f(0, &[]);
        return;
    }
    let num: usize = output_shape.iter().product();
    let mut coord = vec![0usize; rank];
    for linear in 0..num {
        let mut rem = linear;
        for d in (0..rank).rev() {
            coord[d] = rem % output_shape[d];
            rem /= output_shape[d];
        }
        f(linear, &coord);
    }
}

pub(crate) fn build_f32_output(input: &HostData, axis: usize, data: Vec<f32>) -> HostData {
    let out_shape_vec = output_shape(&input.shape, axis);
    let strides = contiguous_strides(&out_shape_vec);
    HostData {
        data: HostDataVec::F32(data),
        shape: Shape::from(out_shape_vec),
        strides,
    }
}

/// `progress` advanced by one output cell, where the caller tracks any.
pub(crate) fn bump(progress: Option<&Progress>) {
    if let Some(progress) = progress {
        progress.bump();
    }
}

/// The values of `input` along `axis` that reduce into the output cell at `out_coord`, in order.
pub(crate) fn row_along<'a>(
    input: &'a HostData,
    axis: usize,
    out_coord: &[usize],
) -> impl Iterator<Item = f32> + 'a {
    let mut coord = out_coord.to_vec();
    (0..input.shape[axis]).map(move |i| {
        coord[axis] = i;
        input.get_f32(&coord)
    })
}

/// The best of `values`, starting from `start`, and its position: a value takes the place of the
/// best so far where `replaces(best, value)`.
pub(crate) fn best_of(
    values: impl Iterator<Item = f32>,
    start: f32,
    replaces: fn(f32, f32) -> bool,
) -> (u32, f32) {
    values
        .enumerate()
        .fold((0, start), |(at, best), (i, value)| {
            if replaces(best, value) {
                (i as u32, value)
            } else {
                (at, best)
            }
        })
}

/// Each row of `input` along `axis` reduced to one value by `reduce_row`, handed the row's values
/// in order: the output of a reduction one cell wide along `axis`.
pub(crate) fn reference_rows(
    input: &HostData,
    axis: usize,
    progress: Option<&Progress>,
    reduce_row: impl Fn(&[f32]) -> f32,
) -> HostData {
    let out_shape = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape.iter().product()];
    for_each_output_coord(&out_shape, |linear, out_coord| {
        let row: Vec<f32> = row_along(input, axis, out_coord).collect();
        data[linear] = reduce_row(&row);
        bump(progress);
    });
    build_f32_output(input, axis, data)
}

/// Each row of `input` along `axis` ranked from its largest value, its first `k` places written
/// along `axis` as `pick` reads each (a value and its index, or none past the row's end).
pub(crate) fn reference_ranked(
    input: &HostData,
    axis: usize,
    k: usize,
    progress: Option<&Progress>,
    pick: impl Fn(Option<&(f32, u32)>) -> f32,
) -> HostData {
    let mut out_shape: Vec<usize> = input.shape.iter().copied().collect();
    out_shape[axis] = k;
    let out_strides = contiguous_strides(&out_shape);
    let mut data = vec![0.0f32; out_shape.iter().product()];
    for_each_output_coord(&output_shape(&input.shape, axis), |_, row_coord| {
        let mut ranked: Vec<(f32, u32)> = row_along(input, axis, row_coord).zip(0u32..).collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        let mut coord = row_coord.to_vec();
        for slot in 0..k {
            coord[axis] = slot;
            let at: usize = coord
                .iter()
                .zip(out_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            data[at] = pick(ranked.get(slot));
            bump(progress);
        }
    });
    HostData {
        data: HostDataVec::F32(data),
        shape: Shape::from(out_shape),
        strides: out_strides,
    }
}
