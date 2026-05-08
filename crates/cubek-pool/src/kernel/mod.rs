pub(crate) mod backward;
pub(crate) mod forward;

use cubecl::{prelude::*, std::FastDivmod};

pub(crate) fn shape_divmod<R: Runtime>(
    binding: &TensorBinding<R>,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

#[cube]
pub(crate) fn decompose_linear(
    index: usize,
    shape: &Sequence<FastDivmod<usize>>,
) -> (usize, usize, usize, usize) {
    let (remainder, c) = shape[3].div_mod(index);
    let (remainder, ow) = shape[2].div_mod(remainder);
    let (remainder, oh) = shape[1].div_mod(remainder);
    let (_, b) = shape[0].div_mod(remainder);

    (b, oh, ow, c)
}
