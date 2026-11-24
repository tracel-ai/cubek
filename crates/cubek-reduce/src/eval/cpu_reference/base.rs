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
use crate::eval::workspace::workspace_for;
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

/// The range a checked input is drawn from.
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
}

impl ReduceInput {
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
/// Where `axis` is not an axis of `shape`, or `input` is a ramp whose shape holds a count of
/// values that is not a power of two at most [`RAMP_MAX_ELEMS`].
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
    let output_handle = unwritten(&client, out_shape, output_dtype);

    let elements = ReduceElements {
        input: input_dtype,
        accumulator: accumulation_dtype,
        out: output_dtype,
    };

    let workspace = workspace_for(&client, &shape, axis, operation, elements, false);
    let outcome = launch_and_capture_outcome(&client, &[&output_handle.handle], |c| {
        kernels::Reduce::new(strategy)
            .with_workspace(workspace)
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
    let values_handle = unwritten(&client, out_shape.clone(), input_dtype);
    let indices_handle = unwritten(&client, out_shape, index_dtype);

    let elements = ReduceElements {
        input: input_dtype,
        accumulator: accumulation_dtype,
        out: input_dtype,
    };

    let workspace = workspace_for(&client, &shape, axis, operation, elements, true);
    let outcome = launch_and_capture_outcome(
        &client,
        &[&values_handle.handle, &indices_handle.handle],
        |c| {
            kernels::Reduce::new(strategy)
                .with_workspace(workspace)
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

/// What an output holds before its launch: a value no comparison's result comes near, so a kernel
/// that writes nothing fails one whose result is zero within its epsilon.
const UNWRITTEN: f32 = 60_000.0;

/// A buffer of `shape` in `dtype`, every cell [`UNWRITTEN`].
fn unwritten(client: &Client, shape: Vec<usize>, dtype: ElemType) -> TensorHandle {
    let cells = shape.iter().product();
    TestInput::builder(client.clone(), shape)
        .dtype(dtype)
        .custom(vec![UNWRITTEN; cells])
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

    Ok(reference(&input_host, axis, operation, progress))
}

/// Number of progress bumps the reduce reference will produce: one per output
/// cell.
///
/// # Panics
///
/// Where `axis` is not an axis of `shape`.
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

/// What `operation` writes for every row of `input` along `axis`, laid as its output is: each
/// row's [`ReduceOperation::reference_cells`] along `axis`.
///
/// # Panics
///
/// Where `axis` is not an axis of `input`, or as [`ReduceOperation::reference_cells`] does.
pub fn reference(
    input: &HostData,
    axis: usize,
    operation: ReduceOperation,
    progress: Option<&Progress>,
) -> HostData {
    let out_shape = output_shape_for(input.shape.as_slice(), axis, operation);
    let out_strides = contiguous_strides(&out_shape);
    let mut rows_shape = out_shape.clone();
    rows_shape[axis] = 1;
    // Every cell is written: a row writes each of its cells.
    let mut data = vec![0.0; out_shape.iter().product()];
    for_each_coord(&rows_shape, |row_coord| {
        let row: Vec<f32> = row_along(input, axis, row_coord).collect();
        let first: usize = row_coord
            .iter()
            .zip(out_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        for (slot, cell) in operation.reference_cells(&row).into_iter().enumerate() {
            data[first + slot * out_strides[axis]] = cell;
            bump(progress);
        }
    });
    HostData {
        data: HostDataVec::F32(data),
        shape: Shape::from(out_shape),
        strides: out_strides,
    }
}

/// `visit` called on every coordinate of `shape`, the last axis fastest.
fn for_each_coord(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    let mut coord = vec![0usize; shape.len()];
    for linear in 0..shape.iter().product() {
        let mut rest = linear;
        for d in (0..shape.len()).rev() {
            coord[d] = rest % shape[d];
            rest /= shape[d];
        }
        visit(&coord);
    }
}

/// `progress` advanced by one output cell, where the caller tracks any.
fn bump(progress: Option<&Progress>) {
    if let Some(progress) = progress {
        progress.bump();
    }
}

/// The values of `input` along `axis` that reduce into the row at `row_coord`, in order.
fn row_along<'a>(
    input: &'a HostData,
    axis: usize,
    row_coord: &[usize],
) -> impl Iterator<Item = f32> + 'a {
    let mut coord = row_coord.to_vec();
    (0..input.shape[axis]).map(move |i| {
        coord[axis] = i;
        input.get_f32(&coord)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value of a ramp lies in `[-1, 1)`.
    #[test]
    fn a_ramp_lies_within_its_range() {
        const ELEMS: usize = 64;
        let ramp = ramp_of_len(ELEMS);
        assert!(ramp.iter().all(|value| (-1.0..1.0).contains(value)));
    }

    /// A ramp holds each of its steps once.
    #[test]
    fn a_ramp_is_a_permutation_of_its_steps() {
        const ELEMS: usize = 64;
        let mut ramp = ramp_of_len(ELEMS);
        ramp.sort_by(f32::total_cmp);
        ramp.dedup();
        assert_eq!(ramp.len(), ELEMS);
    }

    /// A count that is not a power of two is no permutation under the stride.
    #[test]
    #[should_panic]
    fn a_ramp_of_a_count_not_a_power_of_two_panics() {
        ramp_of_len(100);
    }

    /// Past what f16 resolves, a ramp's steps would collide.
    #[test]
    #[should_panic]
    fn a_ramp_finer_than_f16_panics() {
        ramp_of_len(RAMP_MAX_ELEMS * 2);
    }
}
