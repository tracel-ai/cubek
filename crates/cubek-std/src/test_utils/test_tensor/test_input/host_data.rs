use cubecl::{
    CubeElement, TestRuntime, client::ComputeClient, prelude::CubePrimitive,
    std::tensor::TensorHandle,
};

use crate::test_utils::{copy_casted, test_tensor::strides_utils::reorder_by_strides};

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

    pub fn get(&self, i: usize) -> f32 {
        match self {
            HostDataVec::F32(items) => items[i],
            HostDataVec::Bool(_) => panic!("unsupported"),
        }
    }
}

impl HostData {
    pub fn from_tensor_handle(
        client: &ComputeClient<TestRuntime>,
        tensor_handle: &TensorHandle<TestRuntime>,
        host_data_type: HostDataType,
    ) -> Self {
        let shape = tensor_handle.shape.clone();
        let strides = tensor_handle.strides.clone();

        let data = match host_data_type {
            HostDataType::F32 => {
                // Because read_one_tensor rejects non-contiguous strides, we have
                // handle that, if is col major, its strides don't say that
                // Therefore, we must reorder by strides on the received data
                let handle = copy_casted(client, tensor_handle, f32::as_type_native_unchecked());
                let data = f32::from_bytes(&client.read_one_tensor(handle.as_copy_descriptor()))
                    .to_owned();
                let data = reorder_by_strides(&data, &shape, &strides);

                HostDataVec::F32(data)
            }
            HostDataType::Bool => {
                let handle = copy_casted(client, tensor_handle, u8::as_type_native_unchecked());
                let data =
                    u8::from_bytes(&client.read_one_tensor(handle.as_copy_descriptor())).to_owned();
                // Reading the tensor puts it back in row major but we want to keep the original layout
                let data = reorder_by_strides(&data, &shape, &strides);

                HostDataVec::Bool(data.iter().map(|&x| x > 0).collect())
            }
        };

        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        let mut i = 0usize;
        for (d, idx) in index.iter().enumerate() {
            i += idx * self.strides[d];
        }
        self.data.get(i)
    }
}
