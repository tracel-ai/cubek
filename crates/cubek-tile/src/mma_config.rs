//! Host-side load/store method selection for the manual-mma leaf ([`Leaf::Mma`](crate::Leaf)).
//! Ported from cubek-std's `MmaIOConfig`: which fragment transport each role uses is a
//! `(device, storage-type)` decision that queries [`DeviceProperties`], so it is built host-side
//! and carried into the kernel as a comptime value on the [`Leaf`](crate::Leaf) (exactly as the
//! contraction depth `k` is). Both [`space::Leaf`](crate::Leaf) and the instruction leaf
//! ([`MmaData::mma`](crate::MmaData)) read it, so it lives at the crate root rather than beside
//! either.

use cubecl::ir::{DeviceProperties, MatrixIdent, StorageType};

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
        lhs_stage: StorageType,
        rhs_stage: StorageType,
        acc_stage: StorageType,
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

fn load_method(device_props: &DeviceProperties, dtype: StorageType) -> LoadMethod {
    if !matches!(dtype, StorageType::Packed(_, _))
        && device_props.features.matmul.ldmatrix.contains(&dtype)
    {
        LoadMethod::LoadMatrix
    } else {
        LoadMethod::Manual
    }
}

fn store_method(device_props: &DeviceProperties, dtype: StorageType) -> StoreMethod {
    if !matches!(dtype, StorageType::Packed(_, _))
        && device_props.features.matmul.stmatrix.contains(&dtype)
    {
        StoreMethod::StoreMatrix
    } else {
        StoreMethod::Manual
    }
}
