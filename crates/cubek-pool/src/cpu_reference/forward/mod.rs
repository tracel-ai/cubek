mod adaptive_avg_pool;
mod avg_pool;
mod max_pool;

pub use adaptive_avg_pool::run_adaptive_avg_pool;
pub use avg_pool::run_avg_pool;
pub use max_pool::{run_max_pool, run_max_pool_with_indices};

use crate::cpu_reference::decode_index;

pub(crate) fn get_window_coords<const N: usize>(
    spatial_out: &[usize],
    k_coords: &[usize],
    stride: [usize; N],
    padding: [usize; N],
    dilation: [usize; N],
    in_dims: &[usize],
    mut in_coords: Vec<usize>,
) -> Option<Vec<usize>> {
    for d in 0..N {
        let id = spatial_out[d] * stride[d] + k_coords[d] * dilation[d];
        let id_signed = id as isize - padding[d] as isize;

        if id_signed < 0 || id_signed >= in_dims[d + 1] as isize {
            return None;
        }
        in_coords[d + 1] = id_signed as usize;
    }
    Some(in_coords)
}

pub(crate) fn decode_index_simple(index: usize, shape: &[usize]) -> Vec<usize> {
    let strides = row_major_strides_vec(shape);
    decode_index(index, shape, &strides)
}

pub(crate) fn row_major_strides_vec(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
