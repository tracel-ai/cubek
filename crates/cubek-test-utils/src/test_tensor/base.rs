use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::{ElemType, StorageType},
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubecl_common::quant::scheme::QuantScheme;
use cubek_quant::scheme::{QuantLevel, QuantStore};

use crate::test_tensor::{
    arange::build_arange,
    custom::build_custom,
    eye::build_eye,
    host_data::{HostData, HostDataType},
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
            let original_shape = tensor.handle.shape().clone();

            // Derive the scale tensor from the quant value's range. For
            // `QuantLevel::Tensor` a single scale is used; for
            // `QuantLevel::Block` we compute a per-block scale from the host
            // data so each block fully uses the quant range without clipping.
            let (scales_shape, scales_data) = compute_input_scales(&tensor.host, &scheme);
            let scale_handle = TestInput::new(
                client.clone(),
                scales_shape.clone(),
                InputDataType::Standard(f32::as_type_native_unchecked().storage_type()),
                StrideSpec::RowMajor,
                DataKind::Custom { data: scales_data },
            )
            .generate();

            // Determine the correct storage type for the quantized output buffer
            let output_storage_type = match &scheme.store {
                QuantStore::PackedU32(_) => {
                    StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32))
                }
                QuantStore::PackedNative(_) | QuantStore::Native => {
                    StorageType::Scalar(ElemType::from_quant_value(scheme.value))
                }
            };

            let mut quant_shape = original_shape.clone();
            let num_quants = scheme.num_quants();
            // Only divide last dim for PackedU32/PackedNative; Native stores 1:1
            match &scheme.store {
                QuantStore::PackedU32(_) | QuantStore::PackedNative(_) => {
                    if num_quants > 1 {
                        let last_dim = quant_shape.len() - 1;
                        quant_shape[last_dim] /= num_quants;
                    }
                }
                QuantStore::Native => {}
            }

            let output_handle = TestInput::new(
                client.clone(),
                quant_shape,
                InputDataType::Standard(output_storage_type),
                StrideSpec::RowMajor,
                DataKind::Zeros,
            )
            .generate();

            let out_scale_handle = TestInput::new(
                client.clone(),
                scales_shape,
                InputDataType::Standard(f32::as_type_native_unchecked().storage_type()),
                StrideSpec::RowMajor,
                DataKind::Zeros,
            )
            .generate();

            let input_elem = match tensor.handle.dtype {
                StorageType::Scalar(elem) => elem,
                _ => panic!("Unsupported storage type {:?}", tensor.handle.dtype),
            };

            cubek_quant::quantize::launch_ref(
                &client,
                tensor.handle.binding(),
                output_handle.clone().binding(),
                scale_handle.binding(),
                out_scale_handle.clone().binding(),
                &scheme,
                input_elem,
            )
            .expect("Quantization failed");

            // Keep the packed shape on the handle
            tensor.handle = output_handle;
            tensor.quantization = Some(QuantizationInfo {
                scheme,
                scale: out_scale_handle,
                shape: original_shape,
            });
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

/// Compute the scale tensor shape and values for a quantized input.
///
/// For `QuantLevel::Tensor` this returns a single-element scale based on the
/// quant value's range (assumes input in [-1, 1]). For `QuantLevel::Block`
/// each block gets its own scale derived from `max(|value|)` in that block,
/// matching the reference pattern used by the cubek-quant symmetric tests.
fn compute_input_scales(host: &HostData, scheme: &QuantScheme) -> (Shape, Vec<f32>) {
    let (q_min, q_max) = scheme.value.range();
    let max_abs_q = q_max.abs().max(q_min.abs());

    match &scheme.level {
        QuantLevel::Tensor => {
            let shape: Shape = core::iter::repeat_n(1, host.shape.len()).collect();
            (shape, vec![1.0 / max_abs_q])
        }
        QuantLevel::Block(block_size) => {
            let rank = host.shape.len();
            let block_dims: Vec<usize> = block_size
                .to_dim_vec(rank)
                .into_iter()
                .map(|b| b as usize)
                .collect();

            let scales_shape: Shape = host
                .shape
                .iter()
                .zip(block_dims.iter())
                .map(|(d, b)| {
                    assert!(
                        d.is_multiple_of(*b),
                        "Block size {b} must divide dimension {d}",
                    );
                    d / b
                })
                .collect();

            let num_blocks: usize = scales_shape.iter().product();
            let block_elem_count: usize = block_dims.iter().product();

            let mut scales = Vec::with_capacity(num_blocks);
            let mut data_idx = vec![0usize; rank];
            for block_linear in 0..num_blocks {
                // Decode the flat block index into per-dim block indices.
                let mut block_idx = vec![0usize; rank];
                let mut rem = block_linear;
                for d in (0..rank).rev() {
                    block_idx[d] = rem % scales_shape[d];
                    rem /= scales_shape[d];
                }

                let mut block_max = 0.0_f32;
                for elem_linear in 0..block_elem_count {
                    let mut rem = elem_linear;
                    for d in (0..rank).rev() {
                        let within = rem % block_dims[d];
                        data_idx[d] = block_idx[d] * block_dims[d] + within;
                        rem /= block_dims[d];
                    }
                    block_max = block_max.max(host.get_f32(&data_idx).abs());
                }

                // Guard against an all-zero block producing a zero scale that
                // would divide-by-zero inside the quantize kernel.
                let scale = if block_max > 0.0 {
                    block_max / max_abs_q
                } else {
                    1.0 / max_abs_q
                };
                scales.push(scale);
            }

            (scales_shape, scales)
        }
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
