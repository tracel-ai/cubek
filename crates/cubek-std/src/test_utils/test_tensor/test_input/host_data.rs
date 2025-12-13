use cubecl::{
    CubeElement, TestRuntime, client::ComputeClient, prelude::CubePrimitive,
    std::tensor::TensorHandle,
};

use crate::test_utils::copy_casted;

#[derive(Debug)]
pub struct HostData {
    pub data: HostDataVec,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Eq, PartialEq, PartialOrd)]
pub enum HostDataType {
    F32,
    Bool,
}

#[derive(Clone, Debug)]
pub enum HostDataVec {
    F32(Vec<f32>),
    Bool(Vec<bool>),
}

impl HostDataVec {
    pub fn into_f32(self) -> Vec<f32> {
        match self {
            HostDataVec::F32(v) => v,
            _ => panic!("Expected F32 data"),
        }
    }

    pub fn into_bool(self) -> Vec<bool> {
        match self {
            HostDataVec::Bool(v) => v,
            _ => panic!("Expected Bool data"),
        }
    }

    pub fn dtype(&self) -> HostDataType {
        match self {
            HostDataVec::F32(_) => HostDataType::F32,
            HostDataVec::Bool(_) => HostDataType::Bool,
        }
    }
}

impl HostData {
    pub fn from_tensor_handle(
        client: &ComputeClient<TestRuntime>,
        tensor_handle: &TensorHandle<TestRuntime>,
        host_data_type: HostDataType,
    ) -> Self {
        let data = match host_data_type {
            HostDataType::F32 => {
                let handle = copy_casted(client, tensor_handle, f32::as_type_native_unchecked());
                let data = f32::from_bytes(&client.read_one_tensor(handle.as_copy_descriptor()))
                    .to_owned();

                HostDataVec::F32(data)
            }
            HostDataType::Bool => {
                let handle = copy_casted(client, tensor_handle, u8::as_type_native_unchecked());
                let data =
                    u8::from_bytes(&client.read_one_tensor(handle.as_copy_descriptor())).to_owned();

                HostDataVec::Bool(data.iter().map(|&x| x > 0).collect())
            }
        };

        Self {
            data,
            shape: tensor_handle.shape.clone(),
            strides: tensor_handle.strides.clone(),
        }
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        // TODO bad to clone
        let vec = self.data.clone().into_f32();

        let mut i = 0usize;
        for d in 0..index.len() {
            i += index[d] * self.strides[d];
        }

        vec[i]
    }
}
