//! CPU reference and seeded "produce a HostData" primitives for reduce.
//!
//! Both [`strategy_result`] and [`cpu_reference_result`] build the same input
//! bits from `(input, shape, axis, config)` so the two `HostData`s they return
//! are directly comparable for the same `axis`/`config`.

mod all;
mod any;
mod argmax;
mod argmin;
mod argtopk;
mod max;
mod max_abs;
mod mean;
mod min;
mod prod;
mod sum;
mod topk;

pub use all::reference_all;
pub use any::reference_any;
pub use argmax::reference_argmax;
pub use argmin::reference_argmin;
pub use argtopk::reference_argtopk;
pub use max::reference_max;
pub use max_abs::reference_max_abs;
pub use mean::reference_mean;
pub use min::reference_min;
pub use prod::reference_prod;
pub use sum::reference_sum;
pub use topk::reference_topk;

use cubecl::{
    client::Client,
    prelude::*,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, Progress, TestInput,
    launch_and_capture_outcome,
};

use crate::{
    ReduceDtypes, ReduceStrategy, ReduceWithIndicesDtypes,
    components::instructions::ReduceOperationConfig, reduce, reduce_with_indices,
};

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
    /// An evenly spaced ramp over `[-1, 1)`, dealt out by an odd stride so no
    /// run of the tensor is sorted. Every value stays distinct down to f16, so
    /// a top-k has a single right answer and no tie can decide the comparison.
    Ramp,
}

/// Ramp steps f16 still resolves apart: the ramp spans 2.0 and f16 separates
/// neighbours by 2^-11 just below 1.0.
pub const RAMP_MAX_ELEMS: usize = 2048;

/// Odd, so it visits every slot of a power-of-two length.
const RAMP_STRIDE: usize = 1103;

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
            ReduceValues::Uniform(seed) => builder.uniform(seed, -1., 1.),
            ReduceValues::Ramp => builder.custom(ramp(shape.iter().product())),
        }
    }
}

fn ramp(elems: usize) -> Vec<f32> {
    assert!(
        elems.is_power_of_two() && elems <= RAMP_MAX_ELEMS,
        "a ramp of {elems} values is not a permutation, or is finer than f16 resolves"
    );
    (0..elems)
        .map(|i| (i * RAMP_STRIDE % elems) as f32 / elems as f32 * 2.0 - 1.0)
        .collect()
}

/// How far a kernel's output may sit from the reference before it is wrong.
///
/// A selection instruction returns a value taken out of the input, or its
/// coordinate, so any slack at all leaves the comparison unable to fail. Only
/// the accumulating ones round, and over tens of millions of f32 elements some
/// kernels accumulate noticeable noise; tightening those belongs in the
/// per-routine integration tests.
pub fn comparison_epsilon(config: ReduceOperationConfig) -> f32 {
    match config {
        ReduceOperationConfig::Sum | ReduceOperationConfig::Mean | ReduceOperationConfig::Prod => {
            1.0
        }
        ReduceOperationConfig::Max
        | ReduceOperationConfig::Min
        | ReduceOperationConfig::MaxAbs
        | ReduceOperationConfig::ArgMax
        | ReduceOperationConfig::ArgMin
        | ReduceOperationConfig::TopK(_)
        | ReduceOperationConfig::ArgTopK(_)
        | ReduceOperationConfig::Any
        | ReduceOperationConfig::All => 0.0,
    }
}

/// Run `strategy` on `input` and return its output as a [`HostData`].
pub fn strategy_result(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    strategy: ReduceStrategy,
    config: ReduceOperationConfig,
    input: ReduceInput,
) -> Result<HostData, String> {
    let input_dtype = input.dtype;
    let output_dtype = output_dtype_for(&config, input_dtype);
    let accumulation_dtype = f32::elem_type_native();

    let input_handle = input
        .tensor(client.clone(), shape.clone())
        .generate_without_host_data();

    let out_shape = output_shape_for(&shape, axis, &config);
    let output_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(output_dtype)
        .zeros()
        .generate_without_host_data();

    let dtypes = ReduceDtypes {
        input: input_dtype,
        output: output_dtype,
        accumulation: accumulation_dtype,
    };

    let outcome = launch_and_capture_outcome(&client, &[&output_handle.handle], |c| {
        reduce(
            c,
            input_handle.clone().binding(),
            output_handle.clone().binding(),
            axis,
            strategy.clone(),
            config,
            dtypes,
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            output_handle,
            HostDataType::F32,
        )),
    }
}

/// Like [`strategy_result`], but runs the fused `reduce_with_indices` path and
/// returns its **values** output.
///
/// The benchmark catalogue times the fused path, so it has to be able to
/// validate the fused path too: running the plain `reduce` here instead would
/// silently check a kernel that is not the one being measured.
pub fn strategy_result_with_indices(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    strategy: ReduceStrategy,
    config: ReduceOperationConfig,
    input: ReduceInput,
) -> Result<HostData, String> {
    let input_dtype = input.dtype;
    let index_dtype = u32::elem_type_native();
    let accumulation_dtype = f32::elem_type_native();

    let input_handle = input
        .tensor(client.clone(), shape.clone())
        .generate_without_host_data();

    let out_shape = output_shape_for(&shape, axis, &config);
    let values_handle = TestInput::builder(client.clone(), out_shape.clone())
        .dtype(input_dtype)
        .zeros()
        .generate_without_host_data();
    let indices_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(index_dtype)
        .zeros()
        .generate_without_host_data();

    let dtypes = ReduceWithIndicesDtypes {
        input: input_dtype,
        values: input_dtype,
        indices: index_dtype,
        accumulation: accumulation_dtype,
    };

    let outcome = launch_and_capture_outcome(
        &client,
        &[&values_handle.handle, &indices_handle.handle],
        |c| {
            reduce_with_indices(
                c,
                input_handle.clone().binding(),
                values_handle.clone().binding(),
                indices_handle.clone().binding(),
                axis,
                strategy.clone(),
                config,
                dtypes,
            )
            .into()
        },
    );

    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            values_handle,
            HostDataType::F32,
        )),
    }
}

/// CPU-only counterpart to [`strategy_result`]: generate the same seeded
/// inputs, run the matching naive reduce reference, return the result as a
/// [`HostData`].
pub fn cpu_reference_result(
    client: Client,
    shape: Vec<usize>,
    axis: usize,
    config: ReduceOperationConfig,
    input: ReduceInput,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    if let Some(p) = progress {
        let out_shape = output_shape_for(&shape, axis, &config);
        let total: usize = out_shape.iter().product();
        p.set_total(total as u64);
    }

    // A narrow dtype rounds on the way in, so the reference folds what the
    // tensor ended up holding rather than what the generator was handed.
    let (_input_handle, input_host) = input.tensor(client, shape).generate_with_f32_host_data();

    Ok(reference_for_config(&input_host, axis, config, progress))
}

fn reference_for_config(
    input: &HostData,
    axis: usize,
    config: ReduceOperationConfig,
    progress: Option<&Progress>,
) -> HostData {
    match config {
        ReduceOperationConfig::Sum => reference_sum(input, axis, progress),
        ReduceOperationConfig::Mean => reference_mean(input, axis, progress),
        ReduceOperationConfig::Prod => reference_prod(input, axis, progress),
        ReduceOperationConfig::Min => reference_min(input, axis, progress),
        ReduceOperationConfig::Max => reference_max(input, axis, progress),
        ReduceOperationConfig::MaxAbs => reference_max_abs(input, axis, progress),
        ReduceOperationConfig::ArgMax => reference_argmax(input, axis, progress),
        ReduceOperationConfig::ArgMin => reference_argmin(input, axis, progress),
        ReduceOperationConfig::ArgTopK(k) => reference_argtopk(input, axis, k, progress),
        ReduceOperationConfig::TopK(k) => reference_topk(input, axis, k, progress),
        ReduceOperationConfig::Any => reference_any(input, axis, progress),
        ReduceOperationConfig::All => reference_all(input, axis, progress),
    }
}

/// Number of progress bumps the reduce reference will produce: one per output
/// cell.
pub fn cpu_reference_total(shape: &[usize], axis: usize, config: &ReduceOperationConfig) -> u64 {
    let out_shape = output_shape_for(shape, axis, config);
    out_shape.iter().product::<usize>() as u64
}

fn output_shape_for(shape: &[usize], axis: usize, config: &ReduceOperationConfig) -> Vec<usize> {
    let mut out = shape.to_vec();
    out[axis] = match config {
        ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) => *k,
        _ => 1,
    };
    out
}

/// What a `reduce` of `config` writes: a coordinate for the `Arg*` family, the
/// input's own element type for everything else.
pub fn output_dtype_for(config: &ReduceOperationConfig, input_dtype: ElemType) -> ElemType {
    match config {
        ReduceOperationConfig::ArgMax
        | ReduceOperationConfig::ArgMin
        | ReduceOperationConfig::ArgTopK(_) => u32::elem_type_native(),
        _ => input_dtype,
    }
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
