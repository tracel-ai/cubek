//! Host-side configuration: manual-mma load/store selection ([`MmaIo`]) and the software
//! instruction's execution config ([`RegisterBlock`]).

use cubecl::{
    cmma::MatrixIdent,
    ir::{DeviceProperties, ElemType},
};

/// Device-driven choice of load/store methods for a manual-mma tile.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaIo {
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

impl MmaIo {
    /// Select each role's transport from the device's `ldmatrix`/`stmatrix` support.
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

    /// A config forcing the manual path for every role.
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

/// Execution and unrolling configuration for the software instruction.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterBlock {
    /// Scalar register budget for the accumulator block; blocks over budget stay rolled.
    pub budget: usize,
    /// Whether to generate a fast in-bounds path plus checked fallback for edge tiles.
    pub split_edge: bool,
    /// Whether to walk K as (line, component) rather than as a flat scalar walk.
    pub component_fanout: bool,
}

impl RegisterBlock {
    /// A budget with neither specialization turned on.
    pub const fn new(budget: usize) -> Self {
        Self {
            budget,
            split_edge: false,
            component_fanout: false,
        }
    }

    /// Generate the fast in-bounds path plus checked fallback for edge tiles.
    pub const fn split_edge(self) -> Self {
        Self {
            split_edge: true,
            ..self
        }
    }

    /// Walk `K` as (line, component) rather than as a flat scalar walk.
    pub const fn component_fanout(self) -> Self {
        Self {
            component_fanout: true,
            ..self
        }
    }
}
