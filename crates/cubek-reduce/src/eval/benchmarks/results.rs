//! What a launch and the CPU reference each write for one input of a benchmark's problem: both build
//! its bits from the same values, so the two compare directly.

use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::{client::Client, prelude::*};
use cubek_test_utils::{ExecutionOutcome, HostData, HostDataType, Progress, TestInput};

use crate::eval::EvalWorkspace;
use crate::eval::cpu_reference::{ReduceReference, output_shape_for};
use crate::kernels::{self, ReduceElements, ReduceOpStrategy, ReduceWorkspace};
use crate::{ReduceError, ReduceOperation};

/// The bits a comparison runs on: the kernel launch and the CPU reference each
/// build their input from this alone, so the two see the same values.
#[derive(Clone, Copy)]
pub(crate) struct ReduceInput {
    pub dtype: ElemType,
    pub values: ReduceValues,
}

/// Where a comparison's input values come from.
#[derive(Clone, Copy)]
pub(crate) enum ReduceValues {
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
pub(crate) const RAMP_MAX_ELEM_COUNT: usize = 2048;

/// Odd, so it visits every slot of a power-of-two length.
const RAMP_STRIDE_ELEM_COUNT: usize = 1103;

impl ReduceInput {
    /// Uniform values at `dtype`, the input every seeded comparison over a
    /// catalogue-sized shape uses.
    pub(crate) fn uniform(dtype: ElemType, seed: u64) -> Self {
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
            ReduceValues::Ramp => builder.custom(ramp_of_elem_count(shape.iter().product())),
        }
    }
}

/// A ramp of `elem_count` values over `[-1, 1)`, spread by [`RAMP_STRIDE_ELEM_COUNT`] so no run of it
/// is sorted: a permutation of its steps where they are a power of two.
fn ramp_of_elem_count(elem_count: usize) -> Vec<f32> {
    (0..elem_count)
        .map(|i| (i * RAMP_STRIDE_ELEM_COUNT % elem_count) as f32 / elem_count as f32 * 2.0 - 1.0)
        .collect()
}

/// Run `strategy` on `input` and return its values output as a [`HostData`]: by the fused
/// `Reduce::execute_with_indices` path where `with_indices`, so the path a fused benchmark times is
/// the one checked, and by `Reduce::execute` otherwise.
///
/// # Errors
///
/// [`ReduceRunError::Declined`] with the kernel's decline,
/// [`ReduceRunError::OutsideDomain`] where the call lies outside the reduction's Domain, and
/// [`ReduceRunError::DidNotRun`] where the launch the kernel accepted did not run.
///
/// # Panics
///
/// Where `axis` is not an axis of `shape`.
pub(crate) fn strategy_result(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    strategy: ReduceOpStrategy,
    operation: ReduceOperation,
    input: ReduceInput,
    with_indices: bool,
) -> Result<HostData, ReduceRunError> {
    let index_dtype = u32::elem_type_native();
    let out_dtype = match with_indices {
        true => input.dtype,
        false => output_dtype_for(operation, input.dtype),
    };
    let elements = compared_elements(input.dtype, out_dtype);
    let input_handle = input
        .tensor(client.clone(), shape.clone())
        .generate_without_host_data();
    let out_shape = output_shape_for(&shape, axis, operation);
    let values = unwritten(&client, out_shape.clone(), elements.out);
    let indices = unwritten(&client, out_shape, index_dtype);
    let workspace =
        EvalWorkspace::new(&client, &shape, axis, operation, elements, with_indices).workspace;
    let reduce = reduce_given(strategy, workspace);
    let launched = match with_indices {
        true => reduce.execute_with_indices(
            &client,
            input_handle.binding(),
            values.clone().binding(),
            indices.clone().binding(),
            axis,
            operation,
            elements,
            index_dtype,
        ),
        false => reduce.execute(
            &client,
            input_handle.binding(),
            values.clone().binding(),
            axis,
            operation,
            elements,
        ),
    };
    launched.map_err(ReduceRunError::from)?;
    let outcome = ExecutionOutcome::of_outputs(&client, &[&values.handle, &indices.handle]);
    read_back(&client, outcome, values)
}

/// Why a launch the harness compares wrote no output to compare.
#[derive(Debug)]
pub(crate) enum ReduceRunError {
    /// The kernel declined the problem, for the device or as a bug of its own
    /// ([`Decline::Broken`](cubecl::tile::launch::Decline::Broken)).
    Declined(Decline),
    /// The call lay outside the reduction's Domain.
    OutsideDomain(ReduceError),
    /// The launch the kernel accepted did not run.
    DidNotRun(String),
}

impl From<ReduceError> for ReduceRunError {
    fn from(why: ReduceError) -> Self {
        match why {
            ReduceError::Declined(decline) => Self::Declined(decline),
            other => Self::OutsideDomain(other),
        }
    }
}

impl core::fmt::Display for ReduceRunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Declined(decline) => write!(f, "declined: {decline}"),
            Self::OutsideDomain(why) => write!(f, "outside the reduction's Domain: {why}"),
            Self::DidNotRun(why) => write!(f, "did not run: {why}"),
        }
    }
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

/// `handle` read back as f32 where the launch ran, or why it did not run.
fn read_back(
    client: &Client,
    outcome: ExecutionOutcome,
    handle: TensorHandle,
) -> Result<HostData, ReduceRunError> {
    match outcome {
        ExecutionOutcome::CompileError(why) => Err(ReduceRunError::DidNotRun(why)),
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
/// Where `axis` is not an axis of `shape`.
pub(crate) fn cpu_reference_result(
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

    Ok(ReduceReference::new(&input_host, axis, operation, progress).output)
}

/// Number of progress bumps the reduce reference will produce: one per output
/// cell.
fn cpu_reference_total(shape: &[usize], axis: usize, operation: ReduceOperation) -> u64 {
    let out_shape = output_shape_for(shape, axis, operation);
    out_shape.iter().product::<usize>() as u64
}

/// What a `Reduce::execute` of `operation` writes: a coordinate for the `Arg*` family, the
/// input's own element type for everything else.
pub(crate) fn output_dtype_for(operation: ReduceOperation, input_dtype: ElemType) -> ElemType {
    match operation {
        ReduceOperation::ArgMax | ReduceOperation::ArgMin | ReduceOperation::ArgTopK { .. } => {
            u32::elem_type_native()
        }
        _ => input_dtype,
    }
}

/// The accumulator element the harness states for every launch it compares, whatever the input or
/// output element.
pub(crate) fn accumulation_dtype() -> ElemType {
    f32::elem_type_native()
}

/// The elements of a launch the harness compares or times: `input` read, written in `out`,
/// accumulated in [`accumulation_dtype`].
pub(crate) fn compared_elements(input: ElemType, out: ElemType) -> ReduceElements {
    ReduceElements {
        input,
        accumulator: accumulation_dtype(),
        out,
    }
}

/// A reduction by `strategy` given `workspace`, as every launch the harness makes is.
pub(crate) fn reduce_given(
    strategy: ReduceOpStrategy,
    workspace: ReduceWorkspace,
) -> kernels::Reduce {
    kernels::Reduce::new(strategy).with_workspace(workspace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReduceKernel;
    use cubecl::ir::{FloatKind, UIntKind};
    use cubek_test_utils::assert_equals_approx;

    #[test]
    fn a_ramp_lies_within_its_range() {
        const ELEM_COUNT: usize = 64;
        let ramp = ramp_of_elem_count(ELEM_COUNT);
        assert!(ramp.iter().all(|value| (-1.0..1.0).contains(value)));
    }

    #[test]
    fn a_ramp_is_a_permutation_of_its_steps() {
        const ELEM_COUNT: usize = 64;
        let mut ramp = ramp_of_elem_count(ELEM_COUNT);
        ramp.sort_by(f32::total_cmp);
        ramp.dedup();
        assert_eq!(ramp.len(), ELEM_COUNT);
    }

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const SHAPE: [usize; 2] = [4, 16];
    const UNIT: ReduceOpStrategy = ReduceOpStrategy::Specific(ReduceKernel::Unit);

    fn ramp() -> ReduceInput {
        ReduceInput {
            dtype: F32,
            values: ReduceValues::Ramp,
        }
    }

    /// Checks that the unit kernel and the reference write the same values of a [`SHAPE`] reduced
    /// along its rows by `launched`, as the reference reduces by `referenced`.
    fn agrees_with_the_reference(
        launched: ReduceOperation,
        referenced: ReduceOperation,
        with_indices: bool,
    ) {
        let client = cubecl::test_device().client();
        let shape = SHAPE.to_vec();
        let written = strategy_result(
            client.clone(),
            shape.clone(),
            1,
            UNIT,
            launched,
            ramp(),
            with_indices,
        )
        .expect("the unit kernel serves every problem");
        let expected = cpu_reference_result(client, shape, 1, referenced, ramp(), None)
            .expect("the reference reduces every problem");
        assert_equals_approx(&written, &expected, 1e-6)
            .as_test_outcome()
            .enforce();
    }

    #[test]
    fn a_launch_writes_what_the_reference_does() {
        agrees_with_the_reference(ReduceOperation::Max, ReduceOperation::Max, false);
    }

    #[test]
    fn a_fused_launch_gives_the_reference_values() {
        agrees_with_the_reference(
            ReduceOperation::TopK { k: 2 },
            ReduceOperation::TopK { k: 2 },
            true,
        );
    }

    #[test]
    fn a_coordinate_is_written_in_u32() {
        let written = output_dtype_for(ReduceOperation::ArgMax, F32);
        assert_eq!(written, ElemType::UInt(UIntKind::U32));
    }

    #[test]
    fn a_value_is_written_in_the_inputs_element() {
        assert_eq!(output_dtype_for(ReduceOperation::Max, F32), F32);
    }

    #[test]
    fn every_launch_accumulates_in_f32() {
        assert_eq!(accumulation_dtype(), F32);
    }

    #[test]
    fn the_reference_advances_by_a_cell() {
        assert_eq!(
            cpu_reference_total(&SHAPE, 1, ReduceOperation::TopK { k: 2 }),
            8
        );
    }
}
