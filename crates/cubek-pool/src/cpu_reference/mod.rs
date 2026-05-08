use crate::{
    cpu_reference::{
        base::PoolGeometry,
        forward::{row_major_strides_vec, run_adaptive_avg_pool, run_avg_pool, run_max_pool},
    },
    definition::{PoolForwardProblem, PoolMode},
};
use cubecl::zspace::Strides;
use cubek_test_utils::{HostData, HostDataVec};

pub mod base;
mod forward;

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
