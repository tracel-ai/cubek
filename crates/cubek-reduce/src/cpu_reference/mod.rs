//! CPU reference and seeded "produce a HostData" primitives for reduce.
//!
//! Both [`strategy_result`] and [`cpu_reference_result`] build the same input
//! bits from `(seed, shape, axis, config)` so the two `HostData`s they return
//! are directly comparable for the same `axis`/`config`.

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
    TestRuntime,
    client::ComputeClient,
    ir::StorageType,
    prelude::*,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, TestInput, launch_and_capture_outcome,
};

use crate::{
    ReduceDtypes, ReduceStrategy, components::instructions::ReduceOperationConfig, reduce,
};

/// Run `strategy` on a seeded f32 reduce problem and return its output as a
/// [`HostData`].
pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    axis: usize,
    strategy: ReduceStrategy,
    config: ReduceOperationConfig,
    seed: u64,
) -> Result<HostData, String> {
    let input_dtype = f32::as_type_native_unchecked().storage_type();
    let output_dtype = output_dtype_for(&config, input_dtype);
    let accumulation_dtype = f32::as_type_native_unchecked().storage_type();

    let (input_handle, _input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(seed, -1., 1.)
        .generate_with_f32_host_data();

    let out_shape = output_shape_for(&shape, axis, &config);
    let output_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(output_dtype)
        .zeros()
        .generate();

    let dtypes = ReduceDtypes {
        input: input_dtype,
        output: output_dtype,
        accumulation: accumulation_dtype,
    };

    let outcome = launch_and_capture_outcome(&client, |c| {
        reduce::<TestRuntime>(
            c,
            input_handle.clone().binding(),
            output_handle.clone().binding(),
            axis,
            strategy.clone(),
            config,
            dtypes.clone(),
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

/// CPU-only counterpart to [`strategy_result`]: generate the same seeded
/// inputs, run the matching naive reduce reference, return the result as a
/// [`HostData`].
pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    axis: usize,
    config: ReduceOperationConfig,
    seed: u64,
) -> Result<HostData, String> {
    let input_dtype = f32::as_type_native_unchecked().storage_type();

    let (_input_handle, input_host) = TestInput::builder(client.clone(), shape)
        .dtype(input_dtype)
        .uniform(seed, -1., 1.)
        .generate_with_f32_host_data();

    Ok(reference_for_config(&input_host, axis, config))
}

fn reference_for_config(input: &HostData, axis: usize, config: ReduceOperationConfig) -> HostData {
    match config {
        ReduceOperationConfig::Sum => reference_sum(input, axis),
        ReduceOperationConfig::Mean => reference_mean(input, axis),
        ReduceOperationConfig::Prod => reference_prod(input, axis),
        ReduceOperationConfig::Min => reference_min(input, axis),
        ReduceOperationConfig::Max => reference_max(input, axis),
        ReduceOperationConfig::MaxAbs => reference_max_abs(input, axis),
        ReduceOperationConfig::ArgMax => reference_argmax(input, axis),
        ReduceOperationConfig::ArgMin => reference_argmin(input, axis),
        ReduceOperationConfig::ArgTopK(k) => reference_argtopk(input, axis, k),
        ReduceOperationConfig::TopK(k) => reference_topk(input, axis, k),
    }
}

fn output_shape_for(shape: &[usize], axis: usize, config: &ReduceOperationConfig) -> Vec<usize> {
    let mut out = shape.to_vec();
    out[axis] = match config {
        ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) => *k,
        _ => 1,
    };
    out
}

fn output_dtype_for(config: &ReduceOperationConfig, input_dtype: StorageType) -> StorageType {
    match config {
        ReduceOperationConfig::ArgMax
        | ReduceOperationConfig::ArgMin
        | ReduceOperationConfig::ArgTopK(_) => u32::as_type_native_unchecked().storage_type(),
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
