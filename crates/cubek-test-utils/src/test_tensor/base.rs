use cubecl::{
    ir::ElemType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubecl_common::quant::scheme::QuantScheme;

use crate::test_tensor::{
    TestInputBuilder,
    arange::build_arange,
    custom::build_custom,
    eye::build_eye,
    host_data::{HostData, HostDataType},
    quant::apply_quantization,
    random::build_random,
    strides::LayoutSpec,
    zeros::build_zeros,
};

#[derive(Clone)]
/// Information about a quantized tensor in tests.
/// This allows marking a tensor as quantized for the kernel dispatcher
/// while keeping the original unquantized data on the host for reference.
pub struct QuantizationInfo {
    /// The scale tensor on the device.
    pub scale: TensorHandle,
    /// The quantization scheme (e.g., Symmetric, Tensor-wise, etc.)
    pub scheme: QuantScheme,
    /// The original unquantized shape of the tensor.
    pub shape: Shape,
}

#[derive(Clone)]
/// A test tensor which might be marked as quantized.
///
/// This structure couples the device handle, the host reference data,
/// and optional quantization metadata. If `quantization` is `Some`,
/// the handle on the device is expected to contain quantized data
/// (unless it's a dummy quantization for testing purposes).
pub struct TestTensor {
    /// The device handle.
    pub handle: TensorHandle,
    /// The host data, usually stored in f32 for easy reference comparison.
    pub host: HostData,
    /// Optional quantization info.
    pub quantization: Option<QuantizationInfo>,
}

#[derive(Clone, Debug)]
pub enum InputDataType {
    Standard(ElemType),
    Quantized(QuantScheme),
}

impl From<ElemType> for InputDataType {
    fn from(dtype: ElemType) -> Self {
        InputDataType::Standard(dtype)
    }
}

impl InputDataType {
    pub fn elem_type(&self) -> ElemType {
        match self {
            InputDataType::Standard(elem_type) => *elem_type,
            InputDataType::Quantized(_) => todo!(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, InputDataType::Quantized(_))
    }

    pub fn scheme(&self) -> Option<QuantScheme> {
        match self {
            InputDataType::Quantized(scheme) => Some(*scheme),
            _ => None,
        }
    }
}

pub struct TestInput {
    base_spec: BaseInputSpec,
    data_kind: DataKind,
    input_dtype: InputDataType,
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
    /// Start a fluent builder for a test input.
    ///
    /// Defaults: `dtype = f32`, `layout = StridedLayout::RowMajor`. Call
    /// `.dtype(_)` / `.layout(_)` to override, then a finalizer such as
    /// `.arange()`,
    /// `.eye()`, `.zeros()`, `.uniform(seed, lo, hi)`, `.bernoulli(seed, p)`,
    /// or `.custom(data)` to produce a [`TestInput`] ready to generate.
    pub fn builder(client: Client, shape: impl Into<Shape>) -> TestInputBuilder {
        TestInputBuilder::new(client, shape.into())
    }

    pub fn new(
        client: Client,
        shape: impl Into<Shape>,
        dtype: impl Into<InputDataType>,
        layout: impl Into<LayoutSpec>,
        data_kind: DataKind,
    ) -> Self {
        let dtype = dtype.into();
        let storage_type = match &dtype {
            InputDataType::Standard(dtype) => *dtype,
            InputDataType::Quantized(_scheme) => {
                // For quantized input, the initial data is generated as f32 (Standard)
                // then it will be quantized in generate_test_tensor.
                f32::elem_type_native()
            }
        };

        let base_spec = BaseInputSpec {
            client,
            shape: shape.into(),
            dtype: storage_type,
            layout: layout.into(),
        };

        Self {
            base_spec,
            data_kind,
            input_dtype: dtype,
        }
    }

    pub fn generate_with_f32_host_data(self) -> (TensorHandle, HostData) {
        self.generate_with_host_data(HostDataType::F32)
    }

    pub fn generate_with_f64_host_data(self) -> (TensorHandle, HostData) {
        self.generate_with_host_data(HostDataType::F64)
    }

    pub fn generate_with_bool_host_data(self) -> (TensorHandle, HostData) {
        self.generate_with_host_data(HostDataType::Bool)
    }

    pub fn generate_test_tensor(self) -> TestTensor {
        let input_dtype = self.input_dtype.clone();
        let client = self.base_spec.client.clone();
        let (handle, host) = self.generate_with_host_data(HostDataType::F32);

        let mut tensor = TestTensor {
            handle,
            host,
            quantization: None,
        };

        if let InputDataType::Quantized(scheme) = input_dtype {
            apply_quantization(&client, &mut tensor, scheme);
        }

        tensor
    }

    pub fn f32_host_data(self) -> HostData {
        self.generate_with_host_data(HostDataType::F32).1
    }

    pub fn bool_host_data(self) -> HostData {
        self.generate_with_host_data(HostDataType::Bool).1
    }

    // Public API returning only TensorHandle
    pub fn generate_without_host_data(self) -> TensorHandle {
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

    pub fn generate_with_host_data(self, host_data_type: HostDataType) -> (TensorHandle, HostData) {
        let client = self.base_spec.client.clone();
        let tensor_handle = self.generate_without_host_data();
        let host_data =
            HostData::from_tensor_handle(&client, tensor_handle.clone(), host_data_type);
        (tensor_handle, host_data)
    }
}

pub struct BaseInputSpec {
    pub client: Client,
    pub shape: Shape,
    pub dtype: ElemType,
    pub layout: LayoutSpec,
}

impl BaseInputSpec {
    pub(crate) fn strides(&self) -> Strides {
        self.layout.compute_strides(&self.shape)
    }
}

pub struct RandomInputSpec {
    pub seed: u64,
    pub distribution: Distribution,
}

#[derive(Copy, Clone)]
pub enum Distribution {
    /// Uniform random over `[lower, upper]`.
    Uniform(f32, f32),
    /// Bernoulli random with probability `prob` of `1`.
    Bernoulli(f32),
    /// Normal (Gaussian) random with the given `mean` and `std`.
    Normal { mean: f32, std: f32 },
}
