use cubecl::prelude::*;
use cubecl::{self as cubecl};

/// Each unit converts one element: `src` read at its coordinate through its own strides, `dst`
/// written at the same coordinate, laid out contiguous.
#[cube(launch_unchecked)]
pub(crate) fn converted_kernel<S: Float, D: Float>(
    src: &Tensor<S>,
    dst: &mut Tensor<D>,
    elements: u32,
    #[define(S, D)] _types: [ElemType; 2],
) {
    let id = ABSOLUTE_POS;
    if id >= elements as usize {
        terminate!();
    }
    let rank = dst.rank();
    let mut remainder = id;
    let mut src_at = 0;
    for step in 0..rank {
        let axis = rank - 1 - step;
        let coordinate = remainder % dst.shape(axis);
        remainder /= dst.shape(axis);
        src_at += coordinate * src.stride(axis);
    }
    dst[id] = D::cast_from(src[src_at]);
}
