use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::*;
use cubecl::std::{
    FastDivmod,
    tensor::layout::linear::{LinearLayoutLaunch, LinearViewLayoutLaunch},
};

pub mod backward;

pub(crate) fn shape_divmod<R: Runtime>(
    binding: &TensorBinding<R>,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

pub(crate) fn linear_layout<R: Runtime>(
    binding: &TensorBinding<R>,
    vector_size: usize,
) -> LinearLayoutLaunch<R> {
    LinearLayoutLaunch::from_shape_strides(
        binding.shape.clone(),
        binding.strides.clone(),
        // Don't care about type size, only vector size.
        Type::new(StorageType::Scalar(ElemType::UInt(UIntKind::U32))).with_vector_size(vector_size),
        LinearViewLayoutLaunch::new(),
    )
}
