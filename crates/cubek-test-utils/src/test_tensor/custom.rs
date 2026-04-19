use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::BaseInputSpec;
use crate::test_tensor::strides::physical_extent;

fn new_custom_data(
    client: &ComputeClient<TestRuntime>,
    shape: Shape,
    strides: Strides,
    dtype: StorageType,
    contiguous_data: Vec<f32>,
) -> TensorHandle<TestRuntime> {
    let num_logical = shape.iter().product::<usize>();
    assert_eq!(
        contiguous_data.len(),
        num_logical,
        "DataKind::Custom expects data.len() == product(shape)",
    );

    let physical_size = physical_extent(&shape, &strides);

    let mut physical = vec![0.0f32; physical_size];
    scatter_logical_to_physical(&shape, &strides, &contiguous_data, &mut physical);

    let bytes = cast_f32_to_dtype(&physical, dtype);
    let handle = client.create_from_slice(&bytes);

    TensorHandle::new(handle, shape, strides, dtype)
}

/// Scatter logical-indexed `src` values to their physical offsets in `dst`.
///
/// When two logical indices map to the same physical offset (a broadcast
/// stride), the last write wins. Callers that care must supply values that
/// agree across the duplicate indices.
fn scatter_logical_to_physical(shape: &Shape, strides: &Strides, src: &[f32], dst: &mut [f32]) {
    let rank = shape.len();
    let mut coord = vec![0usize; rank];

    for (linear, &value) in src.iter().enumerate() {
        let mut rem = linear;
        for d in (0..rank).rev() {
            coord[d] = rem % shape[d];
            rem /= shape[d];
        }
        let offset: usize = coord.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        dst[offset] = value;
    }
}

fn cast_f32_to_dtype(data: &[f32], dtype: StorageType) -> Vec<u8> {
    match dtype {
        StorageType::Scalar(ElemType::Float(FloatKind::F32)) => f32::as_bytes(data).to_vec(),
        StorageType::Scalar(ElemType::Float(FloatKind::F16)) => {
            let casted: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
            half::f16::as_bytes(&casted).to_vec()
        }
        StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => {
            let casted: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
            half::bf16::as_bytes(&casted).to_vec()
        }
        StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)) => {
            let casted: Vec<u32> = data.iter().map(|&x| x as u32).collect();
            u32::as_bytes(&casted).to_vec()
        }
        StorageType::Scalar(ElemType::Int(cubecl::ir::IntKind::I32)) => {
            let casted: Vec<i32> = data.iter().map(|&x| x as i32).collect();
            i32::as_bytes(&casted).to_vec()
        }
        other => panic!("DataKind::Custom: unsupported storage type {other:?}"),
    }
}

pub(crate) fn build_custom(
    base_spec: BaseInputSpec,
    contiguous_data: Vec<f32>,
) -> TensorHandle<TestRuntime> {
    new_custom_data(
        &base_spec.client,
        base_spec.shape.clone(),
        base_spec.strides(),
        base_spec.dtype,
        contiguous_data,
    )
}
