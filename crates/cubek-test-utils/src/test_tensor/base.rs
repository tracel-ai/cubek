use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::{ElemType, StorageType},
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubecl_common::quant::scheme::QuantScheme;
use cubek_quant::scheme::QuantStore;

use crate::test_tensor::{
    arange::build_arange,
    custom::build_custom,
    eye::build_eye,
    host_data::{HostData, HostDataType},
    quant::apply_quantization,
    random::build_random,
    strides::StrideSpec,
    zeros::build_zeros,
};

#[derive(Clone)]
/// Information about a quantized tensor in tests.
/// This allows marking a tensor as quantized for the kernel dispatcher
/// while keeping the original unquantized data on the host for reference.
pub struct QuantizationInfo {
    /// The scale tensor on the device.
    pub scale: TensorHandle<TestRuntime>,
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
    pub handle: TensorHandle<TestRuntime>,
    /// The host data, usually stored in f32 for easy reference comparison.
    pub host: HostData,
    /// Optional quantization info.
    pub quantization: Option<QuantizationInfo>,
}

#[derive(Clone, Debug)]
pub enum InputDataType {
    Standard(StorageType),
    Quantized(QuantScheme),
}

impl From<StorageType> for InputDataType {
    fn from(dtype: StorageType) -> Self {
        InputDataType::Standard(dtype)
    }
}

impl From<cubecl::ir::ElemType> for InputDataType {
    fn from(elem: cubecl::ir::ElemType) -> Self {
        InputDataType::Standard(StorageType::Scalar(elem))
    }
}

impl InputDataType {
    pub fn storage_type(&self) -> StorageType {
        match self {
            InputDataType::Standard(dtype) => *dtype,
            InputDataType::Quantized(scheme) => {
                let elem = ElemType::from_quant_value(scheme.value);

                match scheme.store {
                    QuantStore::Native => StorageType::Scalar(elem),
                    QuantStore::PackedNative(_) => {
                        // Uses the format's inherent packing factor (e.g., E2M1x2)
                        StorageType::Packed(elem, scheme.native_packing())
                    }
                    QuantStore::PackedU32(_) => {
                        // Usually represents multiple small quants in a 32-bit register
                        // factor would be 4 for 8-bit, 8 for 4-bit, etc.
                        let factor = scheme.num_quants();
                        StorageType::Packed(elem, factor)
                    }
                }
            }
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
    /// Defaults: `dtype = f32`, `stride = RowMajor`. Call `.dtype(_)` /
    /// `.stride(_)` to override, then a finalizer such as `.arange()`,
    /// `.eye()`, `.zeros()`, `.uniform(seed, lo, hi)`, `.bernoulli(seed, p)`,
    /// or `.custom(data)` to produce a [`TestInput`] ready to generate.
    pub fn builder(
        client: ComputeClient<TestRuntime>,
        shape: impl Into<Shape>,
    ) -> TestInputBuilder {
        TestInputBuilder::new(client, shape.into())
    }

    pub fn new(
        client: ComputeClient<TestRuntime>,
        shape: impl Into<Shape>,
        dtype: impl Into<InputDataType>,
        stride_spec: StrideSpec,
        data_kind: DataKind,
    ) -> Self {
        let dtype = dtype.into();
        let storage_type = match &dtype {
            InputDataType::Standard(dtype) => *dtype,
            InputDataType::Quantized(_scheme) => {
                // For quantized input, the initial data is generated as f32 (Standard)
                // then it will be quantized in generate_test_tensor.
                f32::as_type_native_unchecked().storage_type()
            }
        };

        let base_spec = BaseInputSpec {
            client,
            shape: shape.into(),
            dtype: storage_type,
            stride_spec,
        };

        Self {
            base_spec,
            data_kind,
            input_dtype: dtype,
        }
    }

    pub fn generate_with_f32_host_data(self) -> (TensorHandle<TestRuntime>, HostData) {
        self.generate_host_data(HostDataType::F32)
    }

    pub fn generate_with_bool_host_data(self) -> (TensorHandle<TestRuntime>, HostData) {
        self.generate_host_data(HostDataType::Bool)
    }

    pub fn generate_test_tensor(self) -> TestTensor {
        let input_dtype = self.input_dtype.clone();
        let client = self.base_spec.client.clone();
        let (handle, host) = self.generate_with_f32_host_data();

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
    /// Uniform random over `[lower, upper]`.
    Uniform(f32, f32),
    /// Bernoulli random with probability `prob` of `1`.
    Bernoulli(f32),
    /// Normal (Gaussian) random with the given `mean` and `std`.
    Normal { mean: f32, std: f32 },
}

/// Fluent builder for [`TestInput`].
///
/// Use [`TestInput::builder`] to start one. The builder holds the shape,
/// dtype, and stride spec. Call a finalizer (`arange`, `eye`, `zeros`,
/// `uniform`, `bernoulli`, `random`, `custom`) to produce a [`TestInput`]
/// ready to generate a tensor handle, host data, or test tensor.
///
/// # Example
///
/// ```ignore
/// use cubek_test_utils::{TestInput, StrideSpec, Distribution};
///
/// let (handle, host) = TestInput::builder(client, [4, 4])
///     .stride(StrideSpec::ColMajor)
///     .uniform( 0, -1.0, 1.0)
///     .generate_with_f32_host_data();
/// ```
pub struct TestInputBuilder {
    client: ComputeClient<TestRuntime>,
    shape: Shape,
    dtype: Option<InputDataType>,
    stride_spec: StrideSpec,
}

impl TestInputBuilder {
    fn new(client: ComputeClient<TestRuntime>, shape: Shape) -> Self {
        Self {
            client,
            shape,
            dtype: None,
            stride_spec: StrideSpec::RowMajor,
        }
    }

    /// Override the dtype. Defaults to f32.
    pub fn dtype(mut self, dtype: impl Into<InputDataType>) -> Self {
        self.dtype = Some(dtype.into());
        self
    }

    /// Override the stride layout. Defaults to [`StrideSpec::RowMajor`].
    pub fn stride(mut self, stride_spec: StrideSpec) -> Self {
        self.stride_spec = stride_spec;
        self
    }

    fn finalize(self, data_kind: DataKind) -> TestInput {
        let dtype = self.dtype.unwrap_or_else(|| {
            InputDataType::Standard(f32::as_type_native_unchecked().storage_type())
        });
        TestInput::new(self.client, self.shape, dtype, self.stride_spec, data_kind)
    }

    /// `0, 1, 2, …` in row-major order.
    pub fn arange(self) -> TestInput {
        self.finalize(DataKind::Arange { scale: None })
    }

    /// `arange` with each value multiplied by `scale`.
    pub fn arange_scaled(self, scale: f32) -> TestInput {
        self.finalize(DataKind::Arange { scale: Some(scale) })
    }

    /// Identity matrix (1 on the diagonal, 0 elsewhere).
    pub fn eye(self) -> TestInput {
        self.finalize(DataKind::Eye)
    }

    /// All-zeros tensor.
    pub fn zeros(self) -> TestInput {
        self.finalize(DataKind::Zeros)
    }

    /// Random tensor with a custom [`Distribution`].
    pub fn random(self, seed: u64, distribution: Distribution) -> TestInput {
        self.finalize(DataKind::Random { seed, distribution })
    }

    /// Uniform random in `[lo, hi]`.
    pub fn uniform(self, seed: u64, lo: f32, hi: f32) -> TestInput {
        self.random(seed, Distribution::Uniform(lo, hi))
    }

    /// Bernoulli random with probability `p` of 1.
    pub fn bernoulli(self, seed: u64, p: f32) -> TestInput {
        self.random(seed, Distribution::Bernoulli(p))
    }

    /// Normal (Gaussian) random with the given `mean` and `std`.
    pub fn normal(self, seed: u64, mean: f32, std: f32) -> TestInput {
        self.random(seed, Distribution::Normal { mean, std })
    }

    /// Tensor populated from an explicit row-major `Vec<f32>`.
    pub fn custom(self, data: Vec<f32>) -> TestInput {
        self.finalize(DataKind::Custom { data })
    }

    /// Evenly-spaced values from `start` to `end` inclusive, populated in
    /// row-major order. The number of points equals the tensor's element count.
    ///
    /// Equivalent to NumPy's `np.linspace(start, end, num=shape.numel()).reshape(shape)`.
    pub fn linspace(self, start: f32, end: f32) -> TestInput {
        let num_elems: usize = self.shape.iter().product();
        let data = if num_elems == 0 {
            Vec::new()
        } else if num_elems == 1 {
            vec![start]
        } else {
            let step = (end - start) / (num_elems - 1) as f32;
            (0..num_elems).map(|i| start + step * i as f32).collect()
        };
        self.finalize(DataKind::Custom { data })
    }
}
