//! The element types a matmul runs at, as a type-level mapping: one `MatmulPrecision`
//! names the global/stage/register element of each operand. Both architectures launch at
//! a precision, so it sits beside [`MatmulElems`](crate::definition::MatmulElems).

use cubecl::prelude::*;
use half::{bf16, f16};

use crate::definition::MatmulElems;

/// Matrix multiplication precisions.
pub trait MatmulPrecision: Send + Sync + Copy + 'static {
    /// Element type of lhs input tensor of the kernel.
    type Lhs: MatrixPrecision;
    /// Element type of rhs input tensor of the kernel.
    type Rhs: MatrixPrecision;
    /// Element type of acc input tensor of the kernel.
    type Acc: MatrixPrecision;
}

pub trait MatrixPrecision: Send + Sync + Copy + 'static {
    /// Element type of input tensor in global memory
    type Global: Numeric;
    /// Element type once stored in shared memory
    type Stage: Numeric;
    /// Element type once in registers for computation
    type Register: Numeric;
}

impl<EG: Numeric, ES: Numeric> MatrixPrecision for (EG, ES) {
    type Global = EG;
    type Stage = ES;
    type Register = ES;
}

impl<EG: Numeric, ES: Numeric, ER: Numeric> MatrixPrecision for (EG, ES, ER) {
    type Global = EG;
    type Stage = ES;
    type Register = ER;
}

impl MatmulPrecision for f16 {
    type Lhs = (f16, f16);
    type Rhs = (f16, f16);
    #[cfg(target_os = "macos")]
    type Acc = (f16, f16);
    #[cfg(not(target_os = "macos"))]
    type Acc = (f16, f32);
}

impl MatmulPrecision for flex32 {
    type Lhs = (f32, f16);
    type Rhs = (f32, f16);
    type Acc = (f32, f32);
}

impl MatmulPrecision for bf16 {
    type Lhs = (bf16, bf16);
    type Rhs = (bf16, bf16);
    #[cfg(target_os = "macos")]
    type Acc = (bf16, bf16);
    #[cfg(not(target_os = "macos"))]
    type Acc = (bf16, f32);
}

impl MatmulPrecision for f32 {
    type Lhs = (f32, f32);
    type Rhs = (f32, f32);
    type Acc = (f32, f32);
}

impl MatmulPrecision for f64 {
    type Lhs = (f64, f32);
    type Rhs = (f64, f32);
    type Acc = (f64, f32);
}

impl MatmulPrecision for u8 {
    type Lhs = (u8, u8);
    type Rhs = (u8, u8);
    type Acc = (i32, i32);
}

impl MatmulPrecision for u16 {
    type Lhs = (u16, u16);
    type Rhs = (u16, u16);
    type Acc = (i32, i32);
}

impl MatmulPrecision for u32 {
    type Lhs = (u32, u32);
    type Rhs = (u32, u32);
    type Acc = (u32, u32);
}

impl MatmulPrecision for u64 {
    type Lhs = (u64, u64);
    type Rhs = (u64, u64);
    type Acc = (u64, u64);
}

impl MatmulPrecision for i8 {
    type Lhs = (i8, i8);
    type Rhs = (i8, i8);
    type Acc = (i32, i32);
}

impl MatmulPrecision for i16 {
    type Lhs = (i16, i16);
    type Rhs = (i16, i16);
    type Acc = (i32, i32);
}

impl MatmulPrecision for i32 {
    type Lhs = (i32, i32);
    type Rhs = (i32, i32);
    type Acc = (i32, i32);
}

impl MatmulPrecision for i64 {
    type Lhs = (i64, i64);
    type Rhs = (i64, i64);
    type Acc = (i64, i64);
}

impl<Lhs: MatrixPrecision, Rhs: MatrixPrecision, Acc: MatrixPrecision> MatmulPrecision
    for (Lhs, Rhs, Acc)
{
    type Lhs = Lhs;
    type Rhs = Rhs;
    type Acc = Acc;
}

impl MatmulElems {
    pub fn new_deprecated<MP: MatmulPrecision>() -> Self {
        Self {
            lhs_global: <MP::Lhs as MatrixPrecision>::Global::elem_type_native(),
            rhs_global: <MP::Rhs as MatrixPrecision>::Global::elem_type_native(),
            acc_global: <MP::Acc as MatrixPrecision>::Global::elem_type_native(),
            lhs_stage: <MP::Lhs as MatrixPrecision>::Stage::elem_type_native(),
            rhs_stage: <MP::Rhs as MatrixPrecision>::Stage::elem_type_native(),
            acc_stage: <MP::Acc as MatrixPrecision>::Stage::elem_type_native(),
            lhs_register: <MP::Lhs as MatrixPrecision>::Register::elem_type_native(),
            rhs_register: <MP::Rhs as MatrixPrecision>::Register::elem_type_native(),
            acc_register: <MP::Acc as MatrixPrecision>::Register::elem_type_native(),
        }
    }
}
