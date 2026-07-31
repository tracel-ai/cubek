//! Two-level quantization with ue4m3 block scales, the configuration the scheme exists for.
//!
//! Needs native `i8` and e4m3, so it self-skips where either is missing.

use cubecl::{
    features::TypeUsage,
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    server::CopyDescriptor,
    std::tensor::TensorHandle,
    {TestRuntime, zspace::shape},
};
use cubecl_common::e4m3;
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

const M: usize = 32;
const N: usize = 32;
const BLOCK: usize = 32;
const GLOBAL: f32 = 0.01;

/// Block scales that e4m3 represents exactly, so the only error left is the value quantization.
/// A rounded scale would hide an off-by-one in the storage type behind its own tolerance.
const EXACT_SCALES: [f32; 4] = [1.0, 1.5, 2.0, 3.0];

#[test]
fn ue4m3_block_scales_with_an_f32_global_round_trip() {
    let client = TestRuntime::client(&Default::default());
    // Both are needed and they are separate capabilities: `i8` for the native value storage, e4m3
    // for the block scales. The CPU runtime has the first and not the second, so guarding on `i8`
    // alone lets this run somewhere it cannot work.
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion)
        || !e4m3::supported_uses(&client).contains(TypeUsage::Conversion)
    {
        return;
    }

    let shape = shape![M, N];
    let shape_scale = shape![M, N / BLOCK];
    let num_blocks = M * N / BLOCK;

    let scales: Vec<f32> = (0..num_blocks)
        .map(|b| EXACT_SCALES[b % EXACT_SCALES.len()])
        .collect();

    // Each value is an exact integer multiple of its block's effective scale, so a correct kernel
    // reproduces it to the last bit and any dropped or doubled factor shows up immediately.
    let mut data = Vec::with_capacity(M * N);
    for &scale in &scales {
        let effective = GLOBAL * scale;
        for k in 0..BLOCK {
            data.push((k as f32 - 15.0) * effective);
        }
    }

    let input_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());
    let scale_alloc = client.create_tensor_from_slice(
        f32::as_bytes(&scales),
        shape_scale.clone(),
        f32::type_size(),
    );
    let global_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&[GLOBAL]), shape![1], f32::type_size());

    let input = TensorHandle::new(
        input_alloc.memory,
        shape.clone(),
        input_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let scale = TensorHandle::new(
        scale_alloc.memory,
        shape_scale.clone(),
        scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let global = TensorHandle::new(
        global_alloc.memory,
        shape![1],
        global_alloc.strides,
        f32::as_type_native_unchecked(),
    );

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block_tensor([BLOCK as u8], QuantParam::F32))
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native)
        .with_param(QuantParam::UE4M3)
        .with_mode(QuantMode::Symmetric);

    let output = TensorHandle::zeros(&client, shape.clone(), i8::as_type_native_unchecked());
    let output_scale = TensorHandle::zeros(
        &client,
        shape_scale.clone(),
        StorageType::Scalar(ElemType::Float(FloatKind::E4M3)),
    );
    let output_global = TensorHandle::zeros(&client, shape![1], f32::as_type_native_unchecked());
    let output_f = TensorHandle::zeros(&client, shape.clone(), f32::as_type_native_unchecked());

    cubek_quant::quantize::launch_ref(
        &client,
        input.binding(),
        output.clone().binding(),
        scale.binding(),
        Some(global.binding()),
        output_scale.clone().binding(),
        Some(output_global.clone().binding()),
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        output.binding(),
        output_f.clone().binding(),
        output_scale.clone().binding(),
        Some(output_global.clone().binding()),
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    // The block scales have to survive the trip through e4m3 storage. Reading the bytes back
    // catches a wrong element type, which would otherwise only show as a scaling error.
    let stored = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_scale.handle.clone().binding(),
        output_scale.shape().clone(),
        output_scale.strides().clone(),
        1,
    ));
    for (block, &expected) in scales.iter().enumerate() {
        let got = e4m3::from_bits(stored[block]).to_f32();
        assert_eq!(
            got, expected,
            "block {block} scale stored as {got}, expected {expected}"
        );
    }

    let written = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_global.handle.clone().binding(),
        output_global.shape().clone(),
        output_global.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    assert_eq!(f32::from_bytes(&written)[0], GLOBAL);

    let computed = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_f.handle.clone().binding(),
        output_f.shape().clone(),
        output_f.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    let restored = f32::from_bytes(&computed);

    assert_eq!(restored.len(), data.len());
    for (i, (&actual, &expected)) in restored.iter().zip(data.iter()).enumerate() {
        let tolerance = expected.abs() * 1e-5;
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {i}: got {actual}, expected {expected}"
        );
    }
}
