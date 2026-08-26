//! Host-side load/store method selection for the manual-mma form
//! ([`Instruction::Mma`](crate::Instruction)) and execution configuration for the software
//! mma instruction ([`Space::instruction`](crate::Space::instruction)).
//!
//! Ported from cubek-std's `MmaIOConfig`: which fragment transport each role uses is a
//! `(device, storage-type)` decision that queries [`DeviceProperties`], so it is built host-side
//! and carried into the kernel as a comptime value on the operand.s register stage (exactly as
//! the contraction depth `k` is). Both the stage statement and the instruction
//! ([`MmaData::mma`](crate::MmaData)) read it, so it lives at the crate root rather than beside
//! either.

use cubecl::{
    cmma::MatrixIdent,
    ir::{DeviceProperties, ElemType},
};

/// Hardware-capability-driven choice of load/store methods for a manual-mma tile, fixed once per
/// `(device, operand storage types)` and carried by [`Instruction::Mma`](crate::Instruction)
/// because the fragment readers/writers branch on it.
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

/// Execution and unrolling configuration for the software instruction. Every
/// field is stated by the caller: nothing here reads the device, so the same config compiles the
/// same kernel everywhere.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterBlock {
    /// Scalar register budget for the accumulator block. The leaf inlines `mr × nr` vector
    /// cells only while they fit in it; wider lines therefore buy fewer cells, not more
    /// registers. Blocks over budget stay rolled, to avoid spilling.
    pub budget: usize,
    /// Whether to generate a dual-path specialization for masked/edge tiles (fast in-bounds path
    /// plus checked fallback).
    pub split_edge: bool,
    /// Whether to walk K as (line, lane) with fixed comptime extracts, rather than as a flat
    /// scalar walk.
    pub lane_fanout: bool,
}

impl RegisterBlock {
    /// A budget with neither specialization turned on. Both are named at the call site by the
    /// method that turns them on, so a reader never has to open this file to learn what a bare
    /// `true` in a constructor meant.
    pub const fn new(budget: usize) -> Self {
        Self {
            budget,
            split_edge: false,
            lane_fanout: false,
        }
    }

    /// Generate the dual-path specialization for masked edge tiles: a fast path that proves its
    /// reads in bounds once, plus the checked fallback for the instances that straddle an edge.
    pub const fn split_edge(self) -> Self {
        Self {
            split_edge: true,
            ..self
        }
    }

    /// Walk `K` as (line, lane) with fixed comptime extracts, rather than as a flat scalar walk.
    pub const fn lane_fanout(self) -> Self {
        Self {
            lane_fanout: true,
            ..self
        }
    }
}
