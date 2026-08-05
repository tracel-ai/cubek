//! The contract between a two-level scheme and the bindings that serve it.

use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    std::tensor::TensorHandle,
    {TestRuntime, zspace::shape},
};
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

const M: usize = 8;
const N: usize = 32;
const BLOCK: usize = 32;
const GLOBAL: f32 = 1.0 / (127.0 * 448.0);

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
    data: Vec<f32>,
}

fn fixture(scheme: &QuantScheme) -> Fixture {
    // Block scales spanning what a narrow param holds, under a per-tensor scale small enough that
    // neither factor is representable in f16 on its own. Values are exact multiples of the
    // effective scale, so a dropped factor moves the result by orders of magnitude.
    let num_blocks = M * N / BLOCK;
    let scales: Vec<f32> = (0..num_blocks).map(|b| 1.0 + b as f32 * 64.0).collect();

    fixture_with(scheme, GLOBAL, scales, |i| (i % 9) as f32 - 4.0)
}

/// `quant_level` gives the integer each value sits on, so every value is an exact multiple of its
/// block's effective scale and a correct round trip reproduces it.
fn fixture_with(
    scheme: &QuantScheme,
    global: f32,
    scales: Vec<f32>,
    quant_level: impl Fn(usize) -> f32,
) -> Fixture {
    let client = TestRuntime::client(&Default::default());
    let shape = shape![M, N];
    let shape_scale = shape![M, N / BLOCK];

    let data: Vec<f32> = (0..M * N)
        .map(|i| quant_level(i) * global * scales[i / BLOCK])
        .collect();

    let input_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());
    let scale_alloc = client.create_tensor_from_slice(
        f32::as_bytes(&scales),
        shape_scale.clone(),
        f32::type_size(),
    );
    let global_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&[global]), shape![1], f32::type_size());

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
        Some(f.global.clone().binding()),
        f.output_scale.clone().binding(),
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &f.client,
        f.output.binding(),
        out_f16.clone().binding(),
        f.output_scale.binding(),
        Some(f.global.binding()),
        &scheme,
        half::f16::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = f
        .client
        .read_one_unchecked_tensor(out_f16.clone().into_copy_descriptor());
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

/// The effective scale is a tensor magnitude times a block spread, so it reaches further down than
/// the values it scales: here it is subnormal in f16 while every value it produces is an ordinary
/// f16 subnormal. Narrowing the scale before the multiply rounds it to zero and takes the whole
/// tensor with it, so this asserts against the correctly rounded product instead.
#[test]
fn an_effective_scale_below_f16_still_scales_an_f16_output() {
    let scheme = two_level();
    let global = 1e-6;
    let block = 0.01;
    // 1e-8, under half of f16's smallest subnormal, so a narrowed scale is exactly zero.
    let effective = global * block;

    let num_blocks = M * N / BLOCK;
    let f = fixture_with(&scheme, global, vec![block; num_blocks], |i| {
        (i % 255) as f32 - 127.0
    });
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
        Some(f.global.clone().binding()),
        f.output_scale.clone().binding(),
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &f.client,
        f.output.binding(),
        out_f16.clone().binding(),
        f.output_scale.binding(),
        Some(f.global.binding()),
        &scheme,
        half::f16::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = f
        .client
        .read_one_unchecked_tensor(out_f16.clone().into_copy_descriptor());
    let restored = half::f16::from_bytes(&computed);

    // One f16 subnormal step, which is all the room the output type leaves at this magnitude.
    let ulp = half::f16::from_bits(1).to_f32();

    assert_eq!(restored.len(), f.data.len());
    assert!(
        restored.iter().any(|v| v.to_f32() != 0.0),
        "every value came back zero, the effective scale {effective} was flushed"
    );
    for (i, (&actual, &expected)) in restored.iter().zip(f.data.iter()).enumerate() {
        let actual = actual.to_f32();
        assert!(
            (actual - expected).abs() <= ulp,
            "index {i}: got {actual}, expected {expected} within {ulp}"
        );
    }
}
