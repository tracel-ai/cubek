use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};

use crate::test_utils::test_tensor::test_input::{
    arange::build_arange, eye::build_eye, random::build_random, zeros::build_zeros,
};

pub enum TestInput {
    Arange(SimpleInputSpec),
    Eye(SimpleInputSpec),
    Random(RandomInputSpec),
    Zeros(SimpleInputSpec),
}

pub struct SimpleInputSpec {
    pub(crate) client: ComputeClient<TestRuntime>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: StorageType,
    // If None, contiguous is assumed
    pub(crate) strides: Option<Vec<usize>>,
}

pub struct RandomInputSpec {
    pub(crate) inner: SimpleInputSpec,
    pub(crate) seed: u64,
    pub(crate) distribution: Distribution,
}

pub(crate) struct TestInputResult {
    pub handle: TensorHandle<TestRuntime>,
    pub host_data: Option<HostData>,
}

impl SimpleInputSpec {
    pub fn new(client: ComputeClient<TestRuntime>, shape: Vec<usize>, dtype: StorageType) -> Self {
        Self {
            client,
            shape,
            dtype,
            strides: None,
        }
    }

    pub fn with_strides(mut self, strides: Vec<usize>) -> Self {
        self.strides = Some(strides);
        self
    }
}

impl RandomInputSpec {
    pub fn new(
        client: ComputeClient<TestRuntime>,
        shape: Vec<usize>,
        dtype: StorageType,
        seed: u64,
        distribution: Distribution,
    ) -> Self {
        let inner = SimpleInputSpec::new(client, shape, dtype);
        Self {
            inner,
            seed,
            distribution,
        }
    }

    pub fn with_strides(mut self, strides: Vec<usize>) -> Self {
        self.inner = self.inner.with_strides(strides);
        self
    }
}

pub enum HostDataType {
    F32,
    Bool,
}

pub enum HostData {
    F32(Vec<f32>),
    Bool(Vec<bool>),
}

impl HostData {
    pub fn into_f32(self) -> Vec<f32> {
        match self {
            HostData::F32(v) => v,
            _ => panic!("Expected F32 data"),
        }
    }

    pub fn into_bool(self) -> Vec<bool> {
        match self {
            HostData::Bool(v) => v,
            _ => panic!("Expected Bool data"),
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

impl TestInput {
    fn build(self, return_data: Option<HostDataType>) -> Result<TestInputResult, TestInputError> {
        match self {
            TestInput::Arange(spec) => build_arange(spec, return_data),
            TestInput::Eye(spec) => build_eye(spec, return_data),
            TestInput::Random(spec) => build_random(spec, return_data),
            TestInput::Zeros(spec) => build_zeros(spec, return_data),
        }
    }

    pub fn build_with_host_data(
        self,
        host_data_type: HostDataType,
    ) -> Result<(TensorHandle<TestRuntime>, HostData), TestInputError> {
        let mut result = self.build(Some(host_data_type))?;
        match result.host_data.take() {
            Some(data) => Ok((result.handle, data)),
            None => Err(TestInputError::InvalidReturnData),
        }
    }

    pub fn build_without_host_data(self) -> Result<TensorHandle<TestRuntime>, TestInputError> {
        Ok(self.build(None)?.handle)
    }
}
