use cubecl::prelude::*;
use half::{bf16, f16};

use crate::{
    definition::{MatmulElems, MatmulIdent},
    multi_level::components::global::memory::ViewDirection,
};

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

/// Matrix multiplication precisions.
pub trait MatmulTypes: Send + Sync + Copy + 'static {
    /// Element type of lhs input tensor of the kernel.
    type Lhs: MatrixTypes;
    /// Element type of rhs input tensor of the kernel.
    type Rhs: MatrixTypes;
    /// Element type of acc input tensor of the kernel.
    type Acc: MatrixTypes;
}

pub trait MatrixTypes: Send + Sync + Copy + 'static {
    /// Element type of input tensor in global memory
    type Global: Numeric;
    type GlobalSize: Size;
    /// Element type once stored in shared memory
    type Stage: Numeric;
    type StageSize: Size;
    /// Element type once in registers for computation
    type Register: Numeric;
    type RegisterSize: Size;
}

impl<EG: Numeric, SG: Size, ES: Numeric, SS: Size, ER: Numeric, SR: Size> MatrixTypes
    for (EG, SG, ES, SS, ER, SR)
{
    type Global = EG;
    type GlobalSize = SG;
    type Stage = ES;
    type StageSize = SS;
    type Register = ER;
    type RegisterSize = SR;
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

impl<Lhs: MatrixTypes, Rhs: MatrixTypes, Acc: MatrixTypes> MatmulTypes for (Lhs, Rhs, Acc) {
    type Lhs = Lhs;
    type Rhs = Rhs;
    type Acc = Acc;
}

pub type Lhs<MT> = <MT as MatmulTypes>::Lhs;
pub type Rhs<MT> = <MT as MatmulTypes>::Rhs;
pub type Acc<MT> = <MT as MatmulTypes>::Acc;

pub type Global<MT> = <MT as MatrixTypes>::Global;
pub type GlobalSize<MT> = <MT as MatrixTypes>::GlobalSize;

pub type Stage<MT> = <MT as MatrixTypes>::Stage;
pub type StageSize<MT> = <MT as MatrixTypes>::StageSize;

pub type Register<MT> = <MT as MatrixTypes>::Register;
pub type RegisterSize<MT> = <MT as MatrixTypes>::RegisterSize;

// ==================== LHS ====================

// Vector forms
pub type LhsG<MT> = Vector<Global<Lhs<MT>>, GlobalSize<Lhs<MT>>>;
pub type LhsS<MT> = Vector<Stage<Lhs<MT>>, StageSize<Lhs<MT>>>;
pub type LhsR<MT> = Vector<Register<Lhs<MT>>, RegisterSize<Lhs<MT>>>;

// Element / Size splits
pub type LhsGE<MT> = <Lhs<MT> as MatrixTypes>::Global;
pub type LhsGS<MT> = <Lhs<MT> as MatrixTypes>::GlobalSize;

pub type LhsSE<MT> = <Lhs<MT> as MatrixTypes>::Stage;
pub type LhsSS<MT> = <Lhs<MT> as MatrixTypes>::StageSize;

pub type LhsRE<MT> = <Lhs<MT> as MatrixTypes>::Register;
pub type LhsRS<MT> = <Lhs<MT> as MatrixTypes>::RegisterSize;

// ==================== RHS ====================

// Vector forms
pub type RhsG<MT> = Vector<Global<Rhs<MT>>, GlobalSize<Rhs<MT>>>;
pub type RhsS<MT> = Vector<Stage<Rhs<MT>>, StageSize<Rhs<MT>>>;
pub type RhsR<MT> = Vector<Register<Rhs<MT>>, RegisterSize<Rhs<MT>>>;

// Element / Size splits
pub type RhsGE<MT> = <Rhs<MT> as MatrixTypes>::Global;
pub type RhsGS<MT> = <Rhs<MT> as MatrixTypes>::GlobalSize;

pub type RhsSE<MT> = <Rhs<MT> as MatrixTypes>::Stage;
pub type RhsSS<MT> = <Rhs<MT> as MatrixTypes>::StageSize;

pub type RhsRE<MT> = <Rhs<MT> as MatrixTypes>::Register;
pub type RhsRS<MT> = <Rhs<MT> as MatrixTypes>::RegisterSize;

// ==================== ACC ====================

// Vector forms
pub type AccG<MT> = Vector<Global<Acc<MT>>, GlobalSize<Acc<MT>>>;
pub type AccS<MT> = Vector<Stage<Acc<MT>>, StageSize<Acc<MT>>>;
pub type AccR<MT> = Vector<Register<Acc<MT>>, RegisterSize<Acc<MT>>>;

// Element / Size splits
pub type AccGE<MT> = <Acc<MT> as MatrixTypes>::Global;
pub type AccGS<MT> = <Acc<MT> as MatrixTypes>::GlobalSize;

pub type AccSE<MT> = <Acc<MT> as MatrixTypes>::Stage;
pub type AccSS<MT> = <Acc<MT> as MatrixTypes>::StageSize;

pub type AccRE<MT> = <Acc<MT> as MatrixTypes>::Register;
pub type AccRS<MT> = <Acc<MT> as MatrixTypes>::RegisterSize;

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

impl MatmulIdent {
    pub fn view_direction(&self) -> ViewDirection {
        match self {
            MatmulIdent::Lhs => ViewDirection::Col,
            MatmulIdent::Rhs => ViewDirection::Row,
            MatmulIdent::Out => ViewDirection::None,
        }
    }
}
