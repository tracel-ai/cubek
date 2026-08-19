//! Host-side load/store method selection for the manual-mma leaf ([`Leaf::Mma`](crate::Leaf))
//! and execution configuration for the software mma leaf ([`Leaf::Memory`](crate::Leaf)).
//!
//! Ported from cubek-std's `MmaIOConfig`: which fragment transport each role uses is a
//! `(device, storage-type)` decision that queries [`DeviceProperties`], so it is built host-side
//! and carried into the kernel as a comptime value on the [`Leaf`](crate::Leaf) (exactly as the
//! contraction depth `k` is). Both [`space::Leaf`](crate::Leaf) and the instruction leaf
//! ([`MmaData::mma`](crate::MmaData)) read it, so it lives at the crate root rather than beside
//! either.

use cubecl::{
    cmma::MatrixIdent,
    ir::{DeviceProperties, ElemType},
};

/// Hardware-capability-driven choice of load/store methods for a manual-mma tile, fixed once per
/// `(device, operand storage types)` and carried by [`Leaf::Mma`](crate::Leaf) because the
/// fragment readers/writers branch on it.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaIOConfig {
    pub lhs_load_method: LoadMethod,
    pub rhs_load_method: LoadMethod,
    pub acc_load_method: LoadMethod,
    pub store_method: StoreMethod,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadMethod {
    Manual,
    LoadMatrix,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StoreMethod {
    Manual,
    StoreMatrix,
}

impl MmaIOConfig {
    /// Select each role's transport from the device's `ldmatrix`/`stmatrix` support over that
    /// operand's storage element. A packed storage type never uses the intrinsic paths.
    pub fn new(
        device_props: &DeviceProperties,
        lhs_stage: ElemType,
        rhs_stage: ElemType,
        acc_stage: ElemType,
    ) -> Self {
        Self {
            lhs_load_method: load_method(device_props, lhs_stage),
            rhs_load_method: load_method(device_props, rhs_stage),
            acc_load_method: load_method(device_props, acc_stage),
            store_method: store_method(device_props, acc_stage),
        }
    }

    /// A config forcing the manual path for every role: the universal fallback for a backend that
    /// exposes the manual mma but no `ldmatrix`/`stmatrix`, or when the props are not on hand.
    pub fn manual() -> Self {
        Self {
            lhs_load_method: LoadMethod::Manual,
            rhs_load_method: LoadMethod::Manual,
            acc_load_method: LoadMethod::Manual,
            store_method: StoreMethod::Manual,
        }
    }

    pub fn load_method(&self, ident: MatrixIdent) -> LoadMethod {
        match ident {
            MatrixIdent::A => self.lhs_load_method,
            MatrixIdent::B => self.rhs_load_method,
            MatrixIdent::Accumulator => self.acc_load_method,
        }
    }

    pub fn store_method(&self) -> StoreMethod {
        self.store_method
    }
}

fn load_method(device_props: &DeviceProperties, dtype: ElemType) -> LoadMethod {
    if device_props.features.matmul.ldmatrix.contains(&dtype) {
        LoadMethod::LoadMatrix
    } else {
        LoadMethod::Manual
    }
}

fn store_method(device_props: &DeviceProperties, dtype: ElemType) -> StoreMethod {
    if device_props.features.matmul.stmatrix.contains(&dtype) {
        StoreMethod::StoreMatrix
    } else {
        StoreMethod::Manual
    }
}

/// Execution and unrolling configuration for the software (memory/register) MMA leaf. Every
/// field is stated by the caller: nothing here reads the device, so the same config compiles the
/// same kernel everywhere.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MemoryMmaConfig {
    /// Maximum number of vector accumulator cells (mr × nr) to fully inline into registers.
    /// Blocks larger than this remain rolled in loops to avoid register spilling.
    pub unroll_limit: usize,
    /// Whether to generate a dual-path specialization for masked/edge tiles (fast in-bounds path
    /// plus checked fallback).
    pub split_edge: bool,
    /// Whether to walk K as (line, lane) with fixed comptime extracts, rather than as a flat
    /// scalar walk.
    pub lane_fanout: bool,
}

impl MemoryMmaConfig {
    /// The only way to build one: every knob stated, none inferred.
    pub const fn new(unroll_limit: usize, split_edge: bool, lane_fanout: bool) -> Self {
        Self {
            unroll_limit,
            split_edge,
            lane_fanout,
        }
    }
}
