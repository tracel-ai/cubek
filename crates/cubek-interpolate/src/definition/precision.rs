use cubecl::{
    flex32,
    ir::{ElemType, FloatKind},
    prelude::Float,
};

/// Precision used for interpolation.
pub trait InterpolatePrecision: 'static {
    /// Precision used for the input and output tensors.
    type EI: Float;
    /// Precision used for accumulation and weights.
    type EA: Float;
}

impl<EI: Float, EA: Float> InterpolatePrecision for (EI, EA) {
    type EI = EI;
    type EA = EA;
}

impl InterpolatePrecision for f64 {
    type EI = f64;
    type EA = f64;
}

impl InterpolatePrecision for f32 {
    type EI = f32;
    type EA = f32;
}

impl InterpolatePrecision for flex32 {
    type EI = f32;
    type EA = f32;
}

impl InterpolatePrecision for half::f16 {
    type EI = half::f16;
    type EA = f32;
}

impl InterpolatePrecision for half::bf16 {
    type EI = half::bf16;
    type EA = f32;
}

pub fn accumulator_dtype(input: ElemType) -> ElemType {
    match input {
        ElemType::Float(FloatKind::F16)
        | ElemType::Float(FloatKind::BF16)
        | ElemType::Float(FloatKind::Flex32) => ElemType::Float(FloatKind::F32),
        _ => input,
    }
}
