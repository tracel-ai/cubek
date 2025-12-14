use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};

use crate::test_utils::test_tensor::test_input::{
    arange::build_arange,
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
        let spec = RandomInputSpec::new(
            client.clone(),
            shape,
            dtype,
            seed,
            distribution,
            stride_spec,
        );
        TestInput {
            client,
            spec: TestInputSpec::Random(spec),
        }
    }

    pub fn zeros(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
    ) -> Self {
        TestInput {
            client: client.clone(),
            spec: TestInputSpec::Zeros(SimpleInputSpec::new(
                client,
                shape,
                dtype,
                StrideSpec::RowMajor,
            )),
        }
    }

    pub fn eye(client: ComputeClient<TestRuntime>, shape: Vec<usize>, dtype: StorageType) -> Self {
        TestInput {
            client: client.clone(),
            spec: TestInputSpec::Eye(SimpleInputSpec::new(
                client,
                shape,
                dtype,
                StrideSpec::RowMajor,
            )),
        }
    }

    pub fn arange(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        stride_spec: StrideSpec,
    ) -> Self {
        let spec = SimpleInputSpec::new(client.clone(), shape, dtype, stride_spec);

        TestInput {
            client,
            spec: TestInputSpec::Arange(spec),
        }
    }

    pub fn generate_with_f32_host_data(
        self,
    ) -> Result<(TensorHandle<TestRuntime>, HostData), TestInputError> {
        self.generate_with_host_data(HostDataType::F32)
    }

    pub fn generate_with_bool_host_data(
        self,
    ) -> Result<(TensorHandle<TestRuntime>, HostData), TestInputError> {
        self.generate_with_host_data(HostDataType::Bool)
    }

    pub fn generate_without_host_data(self) -> Result<TensorHandle<TestRuntime>, TestInputError> {
        self.generate()
    }

    fn generate(self) -> Result<TensorHandle<TestRuntime>, TestInputError> {
        match self.spec {
            TestInputSpec::Arange(spec) => build_arange(spec),
            TestInputSpec::Eye(spec) => build_eye(spec),
            TestInputSpec::Random(spec) => build_random(spec),
            TestInputSpec::Zeros(spec) => build_zeros(spec),
        }
    }

    fn generate_with_host_data(
        self,
        host_data_type: HostDataType,
    ) -> Result<(TensorHandle<TestRuntime>, HostData), TestInputError> {
        let client = self.client.clone();
        let tensor_handle = self.generate()?;

        let host_data = HostData::from_tensor_handle(&client, &tensor_handle, host_data_type);

        Ok((tensor_handle, host_data))
    }
}

pub struct SimpleInputSpec {
    pub(crate) client: ComputeClient<TestRuntime>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: StorageType,
    pub(crate) stride_spec: StrideSpec,
}

pub struct RandomInputSpec {
    pub(crate) inner: SimpleInputSpec,
    pub(crate) seed: u64,
    pub(crate) distribution: Distribution,
}

impl SimpleInputSpec {
    pub fn new(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        stride_spec: StrideSpec,
    ) -> Self {
        Self {
            client,
            shape,
            dtype,
            stride_spec,
        }
    }
}

impl RandomInputSpec {
    pub fn new(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        seed: u64,
        distribution: Distribution,
        strides: StrideSpec,
    ) -> Self {
        let inner = SimpleInputSpec::new(client, shape, dtype, strides);
        Self {
            inner,
            seed,
            distribution,
        }
    }
}

#[derive(Copy, Clone)]
pub enum Distribution {
    // lower, upper bounds
    Uniform(f32, f32),
    // prob
    Bernoulli(f32),
}

#[derive(Debug)]
pub enum TestInputError {
    UnsupportedStrides,
    InvalidReturnData,
}
