mod backward;
mod forward;
pub mod geometry;

use crate::{
    definition::{
        MaxPoolOptions, PoolBackward, PoolBackwardProblem, PoolForward, PoolForwardProblem,
        PoolMode, PoolProblem,
    },
    eval::cpu_reference::{
        backward::{run_adaptive_avg_pool_backward, run_avg_pool_backward, run_max_pool_backward},
        forward::{
            row_major_strides_vec, run_adaptive_avg_pool, run_avg_pool, run_max_pool,
            run_max_pool_with_indices,
        },
        geometry::PoolGeometry,
    },
};
use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::StorageType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{HostData, HostDataVec, Progress, TestInput};

pub(crate) fn f32_storage_type() -> StorageType {
    f32::as_type_native_unchecked().storage_type()
}

pub(crate) fn i32_storage_type() -> StorageType {
    i32::as_type_native_unchecked().storage_type()
}

pub(crate) fn make_random_f32_host(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    seed: u64,
) -> (TensorHandle<TestRuntime>, HostData) {
    TestInput::builder(client.clone(), shape)
        .uniform(seed, -1., 1.)
        .generate_with_f32_host_data()
}

pub(crate) fn make_zero_handle(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), shape)
        .dtype(dtype)
        .zeros()
        .generate()
}

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: PoolProblem,
    seed: u64,
) -> Result<HostData, String> {
    match problem {
        PoolProblem::Forward(PoolForward::D2(prob)) => forward::strategy_result(client, prob, seed),
        PoolProblem::Forward(_) => Err("cpu reference only supports 2d pool forward".to_string()),
        PoolProblem::Backward(PoolBackward::D2(prob)) => {
            backward::strategy_result(client, prob, seed)
        }
        PoolProblem::Backward(_) => Err("cpu reference only supports 2d pool backward".to_string()),
    }
}

pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    problem: PoolProblem,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    match problem {
        PoolProblem::Forward(PoolForward::D2(prob)) => {
            forward::cpu_reference_result(client, prob, seed, progress)
        }
        PoolProblem::Forward(_) => Err("cpu reference only supports 2d pool forward".to_string()),
        PoolProblem::Backward(PoolBackward::D2(prob)) => {
            backward::cpu_reference_result(client, prob, seed, progress)
        }
        PoolProblem::Backward(_) => Err("cpu reference only supports 2d pool backward".to_string()),
    }
}

pub fn cpu_reference_pool<const N: usize>(
    input: &HostData,
    problem: PoolForwardProblem<N>,
) -> HostData {
    let output_shape_struct = problem.output_shape(&problem.input_shape);
    let out_dims = output_shape_struct.to_vec();
    let in_dims = problem.input_shape.to_vec();

    let out_strides = row_major_strides_vec(&out_dims);

    let output_data = match &problem.mode {
        PoolMode::Max(opts) => run_max_pool(input, opts, &out_dims, &in_dims, &out_strides),
        PoolMode::Avg(opts) => run_avg_pool(input, opts, &out_dims, &in_dims, &out_strides),
        PoolMode::AdaptiveAvg(opts) => {
            run_adaptive_avg_pool(input, opts, &out_dims, &in_dims, &out_strides)
        }
    };

    HostData {
        data: HostDataVec::F32(output_data),
        shape: output_shape_struct,
        strides: Strides::new(&out_strides),
    }
}

pub fn cpu_reference_pool_backward<const N: usize>(
    grad_output: &HostData,
    indices: &HostData,
    problem: PoolBackwardProblem<N>,
) -> HostData {
    let out_dims = grad_output.shape.to_vec();
    let input_shape = Shape::from(vec![
        problem.out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        problem.out_grad_shape[3],
    ]);
    let in_dims = input_shape.to_vec();
    let in_strides = row_major_strides_vec(&in_dims);

    let output_grad = match &problem.mode {
        PoolMode::Max(opts) => {
            if problem.with_indices {
                run_max_pool_backward(grad_output, indices, opts, &in_dims, &out_dims, &in_strides)
            } else {
                unimplemented!("Max pool backward without indices is not implemented yet")
            }
        }
        PoolMode::Avg(_opts) => {
            run_avg_pool_backward(grad_output, _opts, &in_dims, &out_dims, &in_strides)
        }
        PoolMode::AdaptiveAvg(_opts) => {
            run_adaptive_avg_pool_backward(grad_output, _opts, &in_dims, &out_dims, &in_strides)
        }
    };

    HostData {
        data: HostDataVec::F32(output_grad),
        shape: input_shape,
        strides: Strides::new(&row_major_strides_vec(&in_dims)),
    }
}

pub fn cpu_reference_max_pool_indices<const N: usize>(
    input: &HostData,
    opts: &MaxPoolOptions<N>,
    out_dims: &[usize],
) -> HostData {
    let in_dims = input.shape.to_vec();
    let out_strides = row_major_strides_vec(out_dims);
    let (_output, indices) =
        run_max_pool_with_indices(input, opts, out_dims, &in_dims, &out_strides);

    HostData {
        data: HostDataVec::I32(indices),
        shape: Shape::from(out_dims.to_vec()),
        strides: Strides::new(&out_strides),
    }
}

pub(crate) fn decode_index(mut index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for i in 0..shape.len() {
        coords[i] = index / strides[i];
        index %= strides[i];
    }
    coords
}
