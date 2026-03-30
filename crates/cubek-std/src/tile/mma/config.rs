use cubecl::ir::{DeviceProperties, MatrixIdent, StorageType};

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
