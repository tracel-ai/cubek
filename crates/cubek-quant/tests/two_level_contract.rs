//! The contract between a two-level scheme and the bindings that serve it.

use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    server::CopyDescriptor,
    std::tensor::TensorHandle,
    {TestRuntime, zspace::shape},
};
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

const M: usize = 8;
const N: usize = 32;
const BLOCK: usize = 32;
const GLOBAL: f32 = 0.25;

fn two_level() -> QuantScheme {
    QuantScheme::default()
        .with_level(QuantLevel::block_tensor([BLOCK as u8], QuantParam::F32))
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric)
}

struct Fixture {
    client: cubecl::client::ComputeClient<TestRuntime>,
    input: TensorHandle<TestRuntime>,
    scale: TensorHandle<TestRuntime>,
    global: TensorHandle<TestRuntime>,
    output: TensorHandle<TestRuntime>,
    output_scale: TensorHandle<TestRuntime>,
    output_global: TensorHandle<TestRuntime>,
    data: Vec<f32>,
}

fn fixture(scheme: &QuantScheme) -> Fixture {
    let client = TestRuntime::client(&Default::default());
    let shape = shape![M, N];
    let shape_scale = shape![M, N / BLOCK];
    let num_blocks = M * N / BLOCK;

    // Every value is an exact multiple of the effective scale, so a dropped or doubled per-tensor
    // factor moves the result by a factor of four rather than by a rounding step.
    let data: Vec<f32> = (0..M * N)
        .map(|i| ((i % 9) as f32 - 4.0) * GLOBAL)
        .collect();
    let scales = vec![1.0f32; num_blocks];

    let input_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());
    let scale_alloc = client.create_tensor_from_slice(
        f32::as_bytes(&scales),
        shape_scale.clone(),
        f32::type_size(),
    );
    let global_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&[GLOBAL]), shape![1], f32::type_size());

    Fixture {
        input: TensorHandle::new(
            input_alloc.memory,
            shape.clone(),
            input_alloc.strides,
            f32::as_type_native_unchecked(),
        ),
        scale: TensorHandle::new(
            scale_alloc.memory,
            shape_scale.clone(),
            scale_alloc.strides,
            f32::as_type_native_unchecked(),
        ),
        global: TensorHandle::new(
            global_alloc.memory,
            shape![1],
            global_alloc.strides,
            f32::as_type_native_unchecked(),
        ),
        output: TensorHandle::zeros(
            &client,
            shape![M, N / scheme.num_quants()],
            u32::as_type_native_unchecked(),
        ),
        output_scale: TensorHandle::zeros(&client, shape_scale, f32::as_type_native_unchecked()),
        output_global: TensorHandle::zeros(&client, shape![1], f32::as_type_native_unchecked()),
        data,
        client,
    }
}

#[test]
#[should_panic(expected = "requires a per-tensor scale")]
fn quantize_rejects_a_missing_per_tensor_scale() {
    let scheme = two_level();
    let f = fixture(&scheme);

    cubek_quant::quantize::launch_ref(
        &f.client,
        f.input.binding(),
        f.output.binding(),
        f.scale.binding(),
        None,
        f.output_scale.binding(),
        None,
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "does not take a per-tensor scale")]
fn quantize_rejects_an_unexpected_per_tensor_scale() {
    let scheme = two_level().with_level(QuantLevel::block([BLOCK as u8]));
    let f = fixture(&scheme);

    cubek_quant::quantize::launch_ref(
        &f.client,
        f.input.binding(),
        f.output.binding(),
        f.scale.binding(),
        Some(f.global.binding()),
        f.output_scale.binding(),
        Some(f.output_global.binding()),
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "requires a per-tensor scale")]
fn dequantize_rejects_a_missing_per_tensor_scale() {
    let scheme = two_level();
    let f = fixture(&scheme);
    let out = TensorHandle::zeros(&f.client, shape![M, N], f32::as_type_native_unchecked());

    cubek_quant::dequantize::launch_ref(
        &f.client,
        f.output.binding(),
        out.binding(),
        f.output_scale.binding(),
        None,
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();
}

#[test]
fn the_per_tensor_scale_is_read_as_f32_not_as_the_compute_type() {
    let scheme = two_level();
    let f = fixture(&scheme);
    let out_f16 = TensorHandle::zeros(
        &f.client,
        shape![M, N],
        half::f16::as_type_native_unchecked(),
    );

    cubek_quant::quantize::launch_ref(
        &f.client,
        f.input.binding(),
        f.output.clone().binding(),
        f.scale.binding(),
        Some(f.global.binding()),
        f.output_scale.clone().binding(),
        Some(f.output_global.clone().binding()),
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &f.client,
        f.output.binding(),
        out_f16.clone().binding(),
        f.output_scale.binding(),
        Some(f.output_global.binding()),
        &scheme,
        half::f16::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = f.client.read_one_unchecked_tensor(CopyDescriptor::new(
        out_f16.handle.clone().binding(),
        out_f16.shape().clone(),
        out_f16.strides().clone(),
        core::mem::size_of::<half::f16>(),
    ));
    let restored = half::f16::from_bytes(&computed);

    assert_eq!(restored.len(), f.data.len());
    for (i, (&actual, &expected)) in restored.iter().zip(f.data.iter()).enumerate() {
        let actual = actual.to_f32();
        assert!(
            (actual - expected).abs() <= 1e-2,
            "index {i}: got {actual}, expected {expected}"
        );
    }
}
