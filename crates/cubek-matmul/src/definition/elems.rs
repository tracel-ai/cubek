//! The concrete element types a matmul runs at: what each operand is in global
//! memory, in the stage, and in registers.

use cubecl::prelude::*;

use crate::definition::MatmulIdent;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct MatmulElems {
    pub lhs_global: ElemType,
    pub rhs_global: ElemType,
    pub acc_global: ElemType,
    pub lhs_stage: ElemType,
    pub rhs_stage: ElemType,
    pub acc_stage: ElemType,
    pub lhs_register: ElemType,
    pub rhs_register: ElemType,
    pub acc_register: ElemType,
}

#[derive(Clone, Debug)]
pub struct MatmulGlobalElems {
    pub lhs: ElemType,
    pub rhs: ElemType,
    pub out: ElemType,
}

impl MatmulElems {
    pub fn from_globals(global_elems: &MatmulGlobalElems) -> Self {
        let acc_type = if global_elems.out == half::f16::elem_type_native()
            || global_elems.out == half::bf16::elem_type_native()
        {
            f32::elem_type_native()
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

    pub fn from_single_dtype(dtype: ElemType) -> Self {
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

    pub fn global(&self, ident: MatmulIdent) -> ElemType {
        match ident {
            MatmulIdent::Lhs => self.lhs_global,
            MatmulIdent::Rhs => self.rhs_global,
            MatmulIdent::Out => self.acc_global,
        }
    }

    pub fn stage(&self, ident: MatmulIdent) -> ElemType {
        match ident {
            MatmulIdent::Lhs => self.lhs_stage,
            MatmulIdent::Rhs => self.rhs_stage,
            MatmulIdent::Out => self.acc_stage,
        }
    }

    pub fn register(&self, ident: MatmulIdent) -> ElemType {
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
