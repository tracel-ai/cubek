use crate::{
    cpu_reference::PoolGeometry,
    definition::{PoolForwardProblem, PoolMode},
};
use cubecl::zspace::Strides;
use cubek_test_utils::{HostData, HostDataVec};

pub fn cpu_reference_pool<const N: usize>(
    input: &HostData,
    problem: PoolForwardProblem<N>,
) -> HostData {
    let output_shape_struct = problem.output_shape(&problem.input_shape);
    let out_dims = output_shape_struct.to_vec();
    let in_dims = problem.input_shape.to_vec();

    let total_output_elements: usize = out_dims.iter().product();
    let mut output_data = vec![0.0f32; total_output_elements];

    let out_strides = row_major_strides_vec(&out_dims);

    for i in 0..total_output_elements {
        let coords = decode_index(i, &out_dims, &out_strides);
        let batch = coords[0];
        let channel = coords[N + 1];
        let spatial_out = &coords[1..N + 1];

        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        let mut count = 0u32;

        match &problem.mode {
            PoolMode::Max(opts) => {
                let kernel_dims = opts.window.kernel_size;
                let total_k_elems: usize = kernel_dims.iter().product();

                for k_idx in 0..total_k_elems {
                    let k_coords = decode_index_simple(k_idx, &kernel_dims);
                    let mut in_coords = vec![0; N + 2];
                    in_coords[0] = batch;
                    in_coords[N + 1] = channel;

                    let mut is_within = true;
                    for d in 0..N {
                        let id =
                            spatial_out[d] * opts.window.stride[d] + k_coords[d] * opts.dilation[d];
                        let id_signed = id as isize - opts.window.padding[d] as isize;

                        if id_signed < 0 || id_signed >= in_dims[d + 1] as isize {
                            is_within = false;
                            break;
                        }
                        in_coords[d + 1] = id_signed as usize;
                    }

                    if is_within {
                        let val = input.get_f32(&in_coords);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
            }
            PoolMode::Avg(opts) => {
                let kernel_dims = opts.window.kernel_size;
                let total_k_elems: usize = kernel_dims.iter().product();

                let padded_dims: Vec<usize> = (0..N)
                    .map(|d| in_dims[d + 1] + 2 * opts.window.padding[d])
                    .collect();

                for k_idx in 0..total_k_elems {
                    let k_coords = decode_index_simple(k_idx, &kernel_dims);
                    let mut in_coords = vec![0; N + 2];
                    in_coords[0] = batch;
                    in_coords[N + 1] = channel;

                    let mut is_in_input = true;
                    let mut is_in_padded = true;

                    for d in 0..N {
                        let id_pad = spatial_out[d] * opts.window.stride[d] + k_coords[d];
                        let id_signed = id_pad as isize - opts.window.padding[d] as isize;

                        if id_signed < 0 || id_signed >= in_dims[d + 1] as isize {
                            is_in_input = false;
                        } else {
                            in_coords[d + 1] = id_signed as usize;
                        }

                        if id_pad >= padded_dims[d] {
                            is_in_padded = false;
                        }
                    }

                    if is_in_input {
                        sum += input.get_f32(&in_coords);
                        if !opts.count_include_pad {
                            count += 1;
                        }
                    }
                    if opts.count_include_pad && is_in_padded {
                        count += 1;
                    }
                }
            }
            PoolMode::AdaptiveAvg(_) => {
                let mut starts = [0usize; N];
                let mut ends = [0usize; N];
                for d in 0..N {
                    starts[d] = (spatial_out[d] * in_dims[d + 1]) / out_dims[d + 1];
                    ends[d] = ((spatial_out[d] + 1) * in_dims[d + 1])
                        .div_ceil(out_dims[d + 1])
                        .min(in_dims[d + 1]);
                }

                let adaptive_kernel_dims: Vec<usize> =
                    (0..N).map(|d| ends[d] - starts[d]).collect();
                let total_k_elems: usize = adaptive_kernel_dims.iter().product();

                for k_idx in 0..total_k_elems {
                    let k_coords = decode_index_simple(k_idx, &adaptive_kernel_dims);
                    let mut in_coords = vec![0; N + 2];
                    in_coords[0] = batch;
                    in_coords[N + 1] = channel;
                    for d in 0..N {
                        in_coords[d + 1] = starts[d] + k_coords[d];
                    }
                    sum += input.get_f32(&in_coords);
                    count += 1;
                }
            }
        }

        output_data[i] = match &problem.mode {
            PoolMode::Max(_) => {
                if max_val == f32::NEG_INFINITY {
                    0.0
                } else {
                    max_val
                }
            }
            _ => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
        };
    }

    HostData {
        data: HostDataVec::F32(output_data),
        shape: output_shape_struct,
        strides: Strides::new(&row_major_strides_vec(&out_dims)),
    }
}

fn decode_index(mut index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for i in 0..shape.len() {
        coords[i] = index / strides[i];
        index %= strides[i];
    }
    coords
}

fn decode_index_simple(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    for i in 0..shape.len() {
        coords[i] = index / strides[i];
        index %= strides[i];
    }
    coords
}

fn row_major_strides_vec(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
