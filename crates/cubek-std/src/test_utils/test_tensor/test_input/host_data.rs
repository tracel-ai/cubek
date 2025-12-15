use cubecl::{
    CubeElement, TestRuntime, client::ComputeClient, prelude::CubePrimitive,
    std::tensor::TensorHandle,
};

use crate::test_utils::test_tensor::test_input::cast::copy_casted;

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
    pub fn get_f32(&self, i: usize) -> f32 {
        match self {
            HostDataVec::F32(items) => items[i],
            HostDataVec::Bool(_) => panic!("Can't get bool as f32"),
        }
    }

    pub fn get_bool(&self, i: usize) -> bool {
        match self {
            HostDataVec::F32(_) => panic!("Can't get bool as f32"),
            HostDataVec::Bool(items) => items[i],
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
            shape,
            strides,
        }
    }

    pub fn get_f32(&self, index: &[usize]) -> f32 {
        self.data.get_f32(self.strided_index(index))
    }

    pub fn get_bool(&self, index: &[usize]) -> bool {
        self.data.get_bool(self.strided_index(index))
    }

    fn strided_index(&self, index: &[usize]) -> usize {
        let mut i = 0usize;
        for (d, idx) in index.iter().enumerate() {
            i += idx * self.strides[d];
        }
        i
    }
}
