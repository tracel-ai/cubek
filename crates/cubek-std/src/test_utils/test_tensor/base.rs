use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};

use crate::test_utils::test_tensor::{
    arange::build_arange,
    custom::build_custom,
    eye::build_eye,
    host_data::{HostData, HostDataType},
    random::build_random,
    strides::StrideSpec,
    zeros::build_zeros,
};

pub struct TestInput {
    client: ComputeClient<TestRuntime>,
    spec: TestInputSpec,
}

pub enum TestInputSpec {
    Arange(SimpleInputSpec),
    Eye(SimpleInputSpec),
    Random(RandomInputSpec),
    Zeros(SimpleInputSpec),
    Custom(CustomInputSpec),
}

impl TestInput {
    pub fn random(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        seed: u64,
        distribution: Distribution,
        stride_spec: StrideSpec,
    ) -> Self {
        let inner = SimpleInputSpec {
            client: client.clone(),
            shape,
            dtype,
            stride_spec,
        };

        let spec = RandomInputSpec {
            inner,
            seed,
            distribution,
        };

        TestInput {
            client,
            spec: TestInputSpec::Random(spec),
        }
    }

    pub fn zeros(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        stride_spec: StrideSpec,
    ) -> Self {
        TestInput {
            client: client.clone(),
            spec: TestInputSpec::Zeros(SimpleInputSpec {
                client,
                shape,
                dtype,
                stride_spec,
            }),
        }
    }

    pub fn eye(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        stride_spec: StrideSpec,
    ) -> Self {
        TestInput {
            client: client.clone(),
            spec: TestInputSpec::Eye(SimpleInputSpec {
                client,
                shape,
                dtype,
                stride_spec,
            }),
        }
    }

    pub fn arange(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        stride_spec: StrideSpec,
    ) -> Self {
        let spec = SimpleInputSpec {
            client: client.clone(),
            shape,
            dtype,
            stride_spec,
        };

        TestInput {
            client,
            spec: TestInputSpec::Arange(spec),
        }
    }

    pub fn custom(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        stride_spec: StrideSpec,
        data: Vec<f32>,
    ) -> Self {
        let inner = SimpleInputSpec {
            client: client.clone(),
            shape,
            dtype,
            stride_spec,
        };

        let spec = CustomInputSpec { inner, data };

        TestInput {
            client,
            spec: TestInputSpec::Custom(spec),
        }
    }
    pub fn generate_with_f32_host_data(self) -> (TensorHandle<TestRuntime>, HostData) {
        self.generate_host_data(HostDataType::F32)
    }

    pub fn generate_with_bool_host_data(self) -> (TensorHandle<TestRuntime>, HostData) {
        self.generate_host_data(HostDataType::Bool)
    }

    pub fn f32_host_data(self) -> HostData {
        self.generate_host_data(HostDataType::F32).1
    }

    pub fn bool_host_data(self) -> HostData {
        self.generate_host_data(HostDataType::Bool).1
    }

    // Public API returning only TensorHandle
    pub fn generate_without_host_data(self) -> TensorHandle<TestRuntime> {
        self.generate()
    }

    pub fn generate(self) -> TensorHandle<TestRuntime> {
        match self.spec {
            TestInputSpec::Arange(spec) => build_arange(spec),
            TestInputSpec::Eye(spec) => build_eye(spec),
            TestInputSpec::Random(spec) => build_random(spec),
            TestInputSpec::Zeros(spec) => build_zeros(spec),
            TestInputSpec::Custom(spec) => build_custom(spec),
        }
    }

    fn generate_host_data(
        self,
        host_data_type: HostDataType,
    ) -> (TensorHandle<TestRuntime>, HostData) {
        let client = self.client.clone();
        let tensor_handle = self.generate();
        let host_data = HostData::from_tensor_handle(&client, &tensor_handle, host_data_type);
        (tensor_handle, host_data)
    }
}

pub struct SimpleInputSpec {
    pub(crate) client: ComputeClient<TestRuntime>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: StorageType,
    pub(crate) stride_spec: StrideSpec,
}

impl SimpleInputSpec {
    pub(crate) fn strides(&self) -> Vec<usize> {
        self.stride_spec.compute_strides(&self.shape)
    }
}

pub struct RandomInputSpec {
    pub(crate) inner: SimpleInputSpec,
    pub(crate) seed: u64,
    pub(crate) distribution: Distribution,
}

pub struct CustomInputSpec {
    pub(crate) inner: SimpleInputSpec,
    pub(crate) data: Vec<f32>,
}

#[derive(Copy, Clone)]
pub enum Distribution {
    // lower, upper bounds
    Uniform(f32, f32),
    // prob
    Bernoulli(f32),
}
