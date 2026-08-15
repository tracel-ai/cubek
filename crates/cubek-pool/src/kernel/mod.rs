pub(crate) mod backward;
pub(crate) mod forward;

use cubecl::{
    ir::{AddressType, ElemType, FloatKind},
    prelude::*,
    std::FastDivmod,
};

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
pub(crate) fn adaptive_start_index(
    output_size_index: usize,
    output_size: usize,
    input_size: usize,
) -> usize {
    (output_size_index * input_size) / output_size
}

#[cube]
pub(crate) fn adaptive_end_index(
    output_size_index: usize,
    output_size: usize,
    input_size: usize,
) -> usize {
    ((output_size_index + 1) * input_size)
        .div_ceil(output_size)
        .min(input_size)
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

#[cube]
pub(crate) fn decompose_linear_5d(
    index: usize,
    shape: &Sequence<FastDivmod<usize>>,
) -> (usize, usize, usize, usize, usize) {
    let (remainder, c) = shape[4].div_mod(index);
    let (remainder, ow) = shape[3].div_mod(remainder);
    let (remainder, oh) = shape[2].div_mod(remainder);
    let (remainder, od) = shape[1].div_mod(remainder);
    let (_, b) = shape[0].div_mod(remainder);

    (b, od, oh, ow, c)
}

pub(crate) fn accumulator_dtype(input: ElemType) -> ElemType {
    match input {
        ElemType::Float(FloatKind::F16)
        | ElemType::Float(FloatKind::BF16)
        | ElemType::Float(FloatKind::Flex32) => ElemType::Float(FloatKind::F32),
        _ => input,
    }
}

/// Account for intermediate products in adaptive window bounds, not just buffer addresses.
pub(crate) fn adaptive_window_address_type(
    input_size: &[usize],
    output_size: &[usize],
) -> AddressType {
    if input_size.iter().zip(output_size).any(|(input, output)| {
        input
            .checked_mul(*output)
            .is_none_or(|product| product > u32::MAX as usize)
    }) {
        AddressType::U64
    } else {
        AddressType::U32
    }
}
