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

use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::{HostData, HostDataVec};

pub(crate) fn contiguous_strides(shape: &[usize]) -> Strides {
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
