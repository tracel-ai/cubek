use cubecl::prelude::*;
use cubecl::std::FastDivmod;

pub mod bicubic;
pub mod bilinear;
pub mod lanczos3;
pub mod nearest;

pub(crate) fn shape_divmod<R: Runtime>(
    binding: &TensorBinding<R>,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}
