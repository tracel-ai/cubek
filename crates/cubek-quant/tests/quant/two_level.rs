//! Two-level quantization: the contract between the scheme and the bindings that serve it, and
//! the configuration the level exists for.
//!
//! The round trip over every value type and shape lives in `symmetric.rs`. What is here is what
//! only an global scale can get wrong: a missing or unexpected binding, a scale read at the
//! compute type instead of at its own, an effective scale that no longer fits the output type,
//! and inner scales narrow enough to be worth a second level in the first place.

use cubecl::{
    client::ComputeClient,
    features::TypeUsage,
    prelude::*,
    std::tensor::TensorHandle,
    {TestRuntime, zspace::Shape, zspace::shape},
};
use cubecl_common::e4m3;
use cubek_quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};

use super::harness::{dequantize, f32_tensor, quant_outputs, quantize, scale_shape};

const M: usize = 8;
const N: usize = 32;
const BLOCK: usize = 32;
const OUTER: f32 = 1.0 / (127.0 * 448.0);

/// The level under test. `store` and the block dtype are left to the caller: they are what decides
/// whether the block scales are narrow enough to need an global scale at all.
fn two_level(store: QuantStore, block_dtype: ScaleDtype) -> QuantScheme {
    QuantScheme::default()
        .per_block([BLOCK as u8], block_dtype)
        .per_tensor(ScaleDtype::F32)
        .with_value(QuantValue::Q8S)
        .with_store(store)
        .with_mode(QuantMode::Symmetric)
}

/// One level of the same shape, for the tests that hand it a binding it does not take.
fn one_level(store: QuantStore, block_dtype: ScaleDtype) -> QuantScheme {
    QuantScheme::default()
        .per_block([BLOCK as u8], block_dtype)
        .with_value(QuantValue::Q8S)
        .with_store(store)
        .with_mode(QuantMode::Symmetric)
}

/// The scheme for the tests about bindings, where nothing narrows and any mismatch is the
/// binding's own.
fn packed_f32() -> QuantScheme {
    two_level(QuantStore::PackedU32(0), ScaleDtype::F32)
}

struct Fixture {
    client: ComputeClient<TestRuntime>,
    shape: Shape,
    input: TensorHandle<TestRuntime>,
    scale: TensorHandle<TestRuntime>,
    global: TensorHandle<TestRuntime>,
    data: Vec<f32>,
}

fn fixture(scheme: &QuantScheme) -> Fixture {
    // Inner scales spanning what a narrow dtype holds, under an global scale small enough that
    // neither factor is representable in f16 on its own. Values are exact multiples of the
    // effective scale, so a dropped factor moves the result by orders of magnitude.
    let num_blocks = M * N / BLOCK;
    let scales: Vec<f32> = (0..num_blocks).map(|b| 1.0 + b as f32 * 64.0).collect();

    fixture_with(scheme, OUTER, scales, |i| (i % 9) as f32 - 4.0)
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

    let data: Vec<f32> = (0..M * N)
        .map(|i| quant_level(i) * global * scales[i / BLOCK])
        .collect();

    Fixture {
        input: f32_tensor(&client, &data, shape.clone()),
        scale: f32_tensor(&client, &scales, scale_shape(scheme, &shape)),
        global: f32_tensor(&client, &[global], shape![1]),
        shape,
        data,
        client,
    }
}

/// Round trip the fixture into an f16 output, the type narrow enough that a scale formed anywhere
/// but in f32 flushes to zero on the way.
fn round_trip_to_f16(f: &Fixture, scheme: &QuantScheme) -> Vec<half::f16> {
    let (values, scales, out_global) = quantize(
        &f.client,
        scheme,
        &f.input,
        &f.scale,
        Some(&f.global),
        &f.shape,
    );
    let out = dequantize(
        &f.client,
        scheme,
        &values,
        &scales,
        out_global.as_ref(),
        &f.shape,
        half::f16::elem_type_native(),
    );

    let computed = f
        .client
        .read_one_unchecked_tensor(out.into_copy_descriptor());

    half::f16::from_bytes(&computed).to_vec()
}

#[test]
#[should_panic(expected = "takes as many scale bindings, but 1 were provided")]
fn quantize_rejects_a_missing_global_scale() {
    let scheme = packed_f32();
    let f = fixture(&scheme);

    quantize(&f.client, &scheme, &f.input, &f.scale, None, &f.shape);
}

#[test]
#[should_panic(expected = "takes as many scale bindings, but 2 were provided")]
fn quantize_rejects_an_unexpected_global_scale() {
    let scheme = one_level(QuantStore::PackedU32(0), ScaleDtype::F32);
    let f = fixture(&scheme);

    quantize(
        &f.client,
        &scheme,
        &f.input,
        &f.scale,
        Some(&f.global),
        &f.shape,
    );
}

#[test]
#[should_panic(expected = "takes as many scale bindings, but 1 were provided")]
fn dequantize_rejects_a_missing_global_scale() {
    let scheme = packed_f32();
    let f = fixture(&scheme);
    let (values, scales) = quant_outputs(&f.client, &scheme, &f.shape);

    dequantize(
        &f.client,
        &scheme,
        &values,
        &scales,
        None,
        &f.shape,
        f32::elem_type_native(),
    );
}

#[test]
fn the_global_scale_is_read_as_f32_not_as_the_compute_type() {
    let scheme = packed_f32();
    let f = fixture(&scheme);
    let restored = round_trip_to_f16(&f, &scheme);

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
    let scheme = packed_f32();
    let global = 1e-6;
    let block = 0.01;
    // 1e-8, under half of f16's smallest subnormal, so a narrowed scale is exactly zero.
    let effective = global * block;

    let num_blocks = M * N / BLOCK;
    let f = fixture_with(&scheme, global, vec![block; num_blocks], |i| {
        (i % 255) as f32 - 127.0
    });
    let restored = round_trip_to_f16(&f, &scheme);

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

/// ue4m3 inner scales under an f32 global scale, the configuration the level exists for: one byte
/// per block instead of four, with the global scale carrying the range that no longer fits.
///
/// Needs native `i8` and e4m3, so it self-skips where either is missing.
#[test]
fn ue4m3_block_scales_with_an_f32_global_scale_round_trip() {
    // Inner scales e4m3 represents exactly, so the only error left is the value quantization. A
    // rounded scale would hide an off-by-one in the storage type behind its own tolerance.
    const EXACT: [f32; 4] = [1.0, 1.5, 2.0, 3.0];

    let client = TestRuntime::client(&Default::default());
    // Both are needed and they are separate capabilities: `i8` for the native value storage, e4m3
    // for the inner scales. The CPU runtime has the first and not the second, so guarding on `i8`
    // alone lets this run somewhere it cannot work.
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion)
        || !e4m3::supported_uses(&client).contains(TypeUsage::Conversion)
    {
        // Said out loud: this is the only test covering the configuration the level exists for,
        // and on a backend without e4m3 it passes without asserting anything.
        println!("i8 or e4m3 unsupported on this runtime, nothing checked");
        return;
    }

    let scheme = two_level(QuantStore::Native, ScaleDtype::UE4M3);
    let num_blocks = M * N / BLOCK;
    let scales: Vec<f32> = (0..num_blocks).map(|b| EXACT[b % EXACT.len()]).collect();
    let f = fixture_with(&scheme, 0.01, scales.clone(), |i| (i % BLOCK) as f32 - 15.0);

    let (values, stored, out_global) = quantize(
        &f.client,
        &scheme,
        &f.input,
        &f.scale,
        Some(&f.global),
        &f.shape,
    );
    let out = dequantize(
        &f.client,
        &scheme,
        &values,
        &stored,
        out_global.as_ref(),
        &f.shape,
        f32::elem_type_native(),
    );

    // The inner scales have to survive the trip through e4m3 storage. Reading the bytes back
    // catches a wrong element type, which would otherwise only show as a scaling error.
    let bytes = f
        .client
        .read_one_unchecked_tensor(stored.into_copy_descriptor());
    for (block, &expected) in scales.iter().enumerate() {
        let got = e4m3::from_bits(bytes[block]).to_f32();
        assert_eq!(
            got, expected,
            "block {block} scale stored as {got}, expected {expected}"
        );
    }

    let written = f
        .client
        .read_one_unchecked_tensor(f.global.clone().into_copy_descriptor());
    assert_eq!(f32::from_bytes(&written)[0], 0.01);

    let computed = f
        .client
        .read_one_unchecked_tensor(out.into_copy_descriptor());
    let restored = f32::from_bytes(&computed);

    assert_eq!(restored.len(), f.data.len());
    for (i, (&actual, &expected)) in restored.iter().zip(f.data.iter()).enumerate() {
        let tolerance = expected.abs() * 1e-5;
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {i}: got {actual}, expected {expected}"
        );
    }
}
