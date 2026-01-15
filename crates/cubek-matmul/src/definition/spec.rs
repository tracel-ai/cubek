use cubecl::{ir::StorageType, prelude::*};
use half::{bf16, f16};

use crate::definition::MatmulIdent;

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

pub type LhsG<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
pub type LhsS<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Stage;
pub type LhsR<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Register;

pub type RhsG<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
pub type RhsS<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Stage;
pub type RhsR<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Register;

pub type AccG<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Global;
pub type AccS<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Stage;
pub type AccR<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Register;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct MatmulElems {
    pub lhs_global: StorageType,
    pub rhs_global: StorageType,
    pub acc_global: StorageType,
    pub lhs_stage: StorageType,
    pub rhs_stage: StorageType,
    pub acc_stage: StorageType,
    pub lhs_register: StorageType,
    pub rhs_register: StorageType,
    pub acc_register: StorageType,
}

#[derive(Clone, Debug)]
pub struct MatmulGlobalElems {
    pub lhs: StorageType,
    pub rhs: StorageType,
    pub out: StorageType,
}

impl MatmulElems {
    pub fn new_deprecated<MP: MatmulPrecision>() -> Self {
        Self {
            lhs_global: <MP::Lhs as MatrixPrecision>::Global::as_type_native_unchecked(),
            rhs_global: <MP::Rhs as MatrixPrecision>::Global::as_type_native_unchecked(),
            acc_global: <MP::Acc as MatrixPrecision>::Global::as_type_native_unchecked(),
            lhs_stage: <MP::Lhs as MatrixPrecision>::Stage::as_type_native_unchecked(),
            rhs_stage: <MP::Rhs as MatrixPrecision>::Stage::as_type_native_unchecked(),
            acc_stage: <MP::Acc as MatrixPrecision>::Stage::as_type_native_unchecked(),
            lhs_register: <MP::Lhs as MatrixPrecision>::Register::as_type_native_unchecked(),
            rhs_register: <MP::Rhs as MatrixPrecision>::Register::as_type_native_unchecked(),
            acc_register: <MP::Acc as MatrixPrecision>::Register::as_type_native_unchecked(),
        }
    }

    pub fn from_globals(global_elems: &MatmulGlobalElems) -> Self {
        let acc_type = if global_elems.out == half::f16::as_type_native_unchecked()
            || global_elems.out == half::bf16::as_type_native_unchecked()
        {
            f32::as_type_native_unchecked()
        } else {
            global_elems.out
        };

        Self {
            lhs_global: global_elems.lhs,
            rhs_global: global_elems.rhs,
            acc_global: global_elems.out,
            lhs_stage: global_elems.lhs,
            rhs_stage: global_elems.rhs,
            acc_stage: acc_type,
            lhs_register: global_elems.lhs,
            rhs_register: global_elems.rhs,
            acc_register: acc_type,
        }
    }

    pub fn from_single_dtype(dtype: StorageType) -> Self {
        Self {
            lhs_global: dtype,
            rhs_global: dtype,
            acc_global: dtype,
            lhs_stage: dtype,
            rhs_stage: dtype,
            acc_stage: dtype,
            lhs_register: dtype,
            rhs_register: dtype,
            acc_register: dtype,
        }
    }

    pub fn from_define_arrays(
        global: [StorageType; 3],
        stage: [StorageType; 3],
        register: [StorageType; 3],
    ) -> Self {
        Self {
            lhs_global: global[0],
            rhs_global: global[1],
            acc_global: global[2],
            lhs_stage: stage[0],
            rhs_stage: stage[1],
            acc_stage: stage[2],
            lhs_register: register[0],
            rhs_register: register[1],
            acc_register: register[2],
        }
    }

    pub fn global(&self, ident: MatmulIdent) -> StorageType {
        match ident {
            MatmulIdent::Lhs => self.lhs_global,
            MatmulIdent::Rhs => self.rhs_global,
            MatmulIdent::Out => self.acc_global,
        }
    }

    pub fn stage(&self, ident: MatmulIdent) -> StorageType {
        match ident {
            MatmulIdent::Lhs => self.lhs_stage,
            MatmulIdent::Rhs => self.rhs_stage,
            MatmulIdent::Out => self.acc_stage,
        }
    }

    pub fn register(&self, ident: MatmulIdent) -> StorageType {
        match ident {
            MatmulIdent::Lhs => self.lhs_register,
            MatmulIdent::Rhs => self.rhs_register,
            MatmulIdent::Out => self.acc_register,
        }
    }

    pub fn as_global_elems(&self) -> MatmulGlobalElems {
        MatmulGlobalElems {
            lhs: self.lhs_global,
            rhs: self.rhs_global,
            out: self.acc_global,
        }
    }

    /// Prefer output type for stage because it's the same size at best, but often smaller.
    /// Having stage == global also enables things like TMA, and an f16 stage for output enables
    /// using `stmatrix` on the registers after casting.
    pub fn adjust_stage_dtypes(&mut self) {
        self.lhs_stage = self.lhs_global;
        self.rhs_stage = self.rhs_global;
        self.acc_stage = self.acc_global;
    }
}
