mod backward;
mod forward;
pub mod geometry;

use crate::{
    cpu_reference::{
        backward::{run_adaptive_avg_pool_backward, run_avg_pool_backward, run_max_pool_backward},
        forward::{
            row_major_strides_vec, run_adaptive_avg_pool, run_avg_pool, run_max_pool,
            run_max_pool_with_indices,
        },
        geometry::PoolGeometry,
    },
    definition::{MaxPoolOptions, PoolBackwardProblem, PoolForwardProblem, PoolMode},
};
use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::{HostData, HostDataVec};

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
