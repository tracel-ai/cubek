use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::StorageType,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::test_tensor::{
    arange::build_arange,
    custom::build_custom,
    eye::build_eye,
    host_data::{HostData, HostDataType},
    random::build_random,
    strides::StrideSpec,
    zeros::build_zeros,
};

pub struct TestInput {
    base_spec: BaseInputSpec,
    data_kind: DataKind,
}

pub enum DataKind {
    Arange {
        scale: Option<f32>,
    },
    Eye,
    Zeros,
    Random {
        seed: u64,
        distribution: Distribution,
    },
    Custom {
        data: Vec<f32>,
    },
}

impl TestInput {
    pub fn new(
        client: ComputeClient<TestRuntime>,
        shape: impl Into<Shape>,
        dtype: StorageType,
        stride_spec: StrideSpec,
        data_kind: DataKind,
    ) -> Self {
        let base_spec = BaseInputSpec {
            client,
            shape: shape.into(),
            dtype,
            stride_spec,
        };

        Self {
            base_spec,
            data_kind,
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
        let (shape, strides, dtype) = (
            self.base_spec.shape.clone(),
            self.base_spec.strides(),
            self.base_spec.dtype,
        );

        let mut handle = match self.data_kind {
            DataKind::Arange { scale } => build_arange(self.base_spec, scale),
            DataKind::Eye => build_eye(self.base_spec),
            DataKind::Random { seed, distribution } => {
                build_random(self.base_spec, seed, distribution)
            }
            DataKind::Zeros => build_zeros(self.base_spec),
            DataKind::Custom { data } => build_custom(self.base_spec, data),
        };
        handle.metadata.shape = shape;
        handle.metadata.strides = strides;
        handle.dtype = dtype;

        handle
    }

    fn generate_host_data(
        self,
        host_data_type: HostDataType,
    ) -> (TensorHandle<TestRuntime>, HostData) {
        let client = self.base_spec.client.clone();

        let tensor_handle = self.generate();
        let host_data =
            HostData::from_tensor_handle(&client, tensor_handle.clone(), host_data_type);

        (tensor_handle, host_data)
    }
}

pub struct BaseInputSpec {
    pub client: ComputeClient<TestRuntime>,
    pub shape: Shape,
    pub dtype: StorageType,
    pub stride_spec: StrideSpec,
}

impl BaseInputSpec {
    pub(crate) fn strides(&self) -> Strides {
        self.stride_spec.compute_strides(&self.shape)
    }
}

pub struct RandomInputSpec {
    pub seed: u64,
    pub distribution: Distribution,
}

#[derive(Copy, Clone)]
pub enum Distribution {
    // lower, upper bounds
    Uniform(f32, f32),
    // prob
    Bernoulli(f32),
}
