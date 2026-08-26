//! `e4m3` and `e5m2`: that an fp8 tensor reconstructs on the *format's* grid rather than on the
//! integers of the same width, packed into `u32` words or stored natively.
//!
//! The same question `fp4.rs` asks, and it belongs here for the same reason: a stored value is the
//! format's *code*, and only for the integer value types is a code the same number as the magnitude
//! it stands for. `e2m1` is where that first bites, but `e4m3` is where it bites hardest — its
//! codes run to ±448 while a byte read as two's complement stops at ±127, so a codec that confuses
//! the two does not merely lose precision, it wraps: `300.0` lands on `44.0`.
//!
//! Both stores are asked, because they arrive at the grid by different routes and only the packed
//! one is software throughout. The native store hands the backend an fp8 element and lets the
//! conversion happen there — but the *scaling* ahead of it is shared, and a quantizer that rounded
//! to whole numbers first would land on the integer grid however faithful the conversion after it.

use cubecl::{
    features::TypeUsage,
    prelude::*,
    {TestRuntime, zspace::shape},
};
use cubek_quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};

use super::harness::{dequantize, f32_tensor, quantize};

/// One block, sized to the vector the packed path builds.
const BLOCK: usize = 8;

fn scheme(value: QuantValue, store: QuantStore) -> QuantScheme {
    QuantScheme::default()
        .with_value(value)
        .with_store(store)
        .with_mode(QuantMode::Symmetric)
        .per_block([BLOCK as u8], ScaleDtype::F32)
}

/// Round trip `data` at an exact scale of one, so the reconstruction is the codec's answer and
/// nothing else.
fn round_trip(value: QuantValue, data: &[f32]) -> Vec<f32> {
    round_trip_stored(value, QuantStore::PackedU32(0), data)
}

fn round_trip_stored(value: QuantValue, store: QuantStore, data: &[f32]) -> Vec<f32> {
    let client = TestRuntime::client(&Default::default());
    let scheme = scheme(value, store);
    let shape = shape![1, BLOCK];

    let input = f32_tensor(&client, data, shape.clone());
    let scale = f32_tensor(&client, &[1.0], shape![1, 1]);

    let (values, scales, global) = quantize(&client, &scheme, &input, &scale, None, &shape);
    let out = dequantize(
        &client,
        &scheme,
        &values,
        &scales,
        global.as_ref(),
        &shape,
        f32::elem_type_native(),
    );

    let computed = client.read_one_unchecked_tensor(out.into_copy_descriptor());
    f32::from_bytes(&computed).to_vec()
}

/// Values `e4m3` holds exactly come back exactly, fractions included. An integer codec fails this
/// on every one below `1.0` and on the halves above it.
#[test]
fn e4m3_code_points_survive_the_round_trip() {
    let data = [0.5f32, 1.5, 2.5, -0.5, -1.5, 0.0625, 288.0, -288.0];
    let computed = round_trip(QuantValue::E4M3, &data);

    assert_eq!(
        computed, data,
        "an e4m3 code point did not reconstruct as itself"
    );
}

/// The same for `e5m2`, whose wider exponent reaches further at coarser steps.
#[test]
fn e5m2_code_points_survive_the_round_trip() {
    let data = [0.5f32, 1.5, 3.0, -0.5, -1.5, 0.0625, 49152.0, -49152.0];
    let computed = round_trip(QuantValue::E5M2, &data);

    assert_eq!(
        computed, data,
        "an e5m2 code point did not reconstruct as itself"
    );
}

/// Magnitudes above what a byte read as two's complement can hold. A field cast from the value
/// rather than encoded wraps here, landing on an unrelated magnitude — which is the failure that
/// costs accuracy silently, since nothing about it raises an error.
#[test]
fn large_magnitudes_do_not_wrap_out_of_the_field() {
    // 300 is not an e4m3 value; 288 and 320 are its neighbours and 304 the midpoint, so it rounds
    // down to 288. What matters is that it lands near itself rather than a wrap away from it.
    let data = [300.0f32, -300.0, 448.0, -448.0, 256.0, -256.0, 130.0, 200.0];
    let computed = round_trip(QuantValue::E4M3, &data);

    for (input, got) in data.iter().zip(&computed) {
        assert!(
            got.signum() == input.signum() && (got - input).abs() <= input.abs() * 0.0625,
            "{input} reconstructed as {got}"
        );
    }
}

/// Past the top of the format, where the codec saturates rather than reaching an infinity or
/// wrapping. `e4m3`'s maximum is 448.
#[test]
fn magnitudes_past_the_maximum_saturate() {
    let data = [1e4f32, -1e4, 1e30, -1e30, 449.0, -449.0, 448.0, -448.0];
    let computed = round_trip(QuantValue::E4M3, &data);

    for (input, got) in data.iter().zip(&computed) {
        assert_eq!(
            *got,
            448.0 * input.signum(),
            "{input} did not saturate at the format's maximum"
        );
    }
}

/// Whether this runtime can hold a natively stored fp8 value, which is the same question the
/// launch asks before it dispatches: an 8-bit float is a byte wherever there is one at all, and a
/// backend with no 8-bit scalar type has nowhere to put one. WGSL is such a backend — it packs fp8
/// four lanes to a `u32` — so these tests sit out there, as the `ue8m0` ones in `fp4.rs` do.
fn native_store_is_addressable() -> bool {
    let client = TestRuntime::client(&Default::default());
    if i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return true;
    }
    println!("no 8-bit type on this runtime, a natively stored fp8 value has nowhere to live");
    false
}

/// The native store, where the value is written as an fp8 element rather than as a field in a
/// word. Nothing in the packed tests reaches this path — it takes its own kernel — and the scaling
/// ahead of the conversion is what these fix: a quantizer that rounded to whole numbers first would
/// hand the format `1.0` where the value was `0.5`, and no faithfulness in the conversion after it
/// could put the fraction back.
#[test]
fn e4m3_code_points_survive_a_native_round_trip() {
    if !native_store_is_addressable() {
        return;
    }

    let data = [0.5f32, 1.5, 2.5, -0.5, -1.5, 0.0625, 288.0, -288.0];
    let computed = round_trip_stored(QuantValue::E4M3, QuantStore::Native, &data);

    assert_eq!(
        computed, data,
        "an e4m3 code point did not reconstruct as itself in the native store"
    );
}

/// The same for `e5m2` natively stored.
#[test]
fn e5m2_code_points_survive_a_native_round_trip() {
    if !native_store_is_addressable() {
        return;
    }

    let data = [0.5f32, 1.5, 3.0, -0.5, -1.5, 0.0625, 49152.0, -49152.0];
    let computed = round_trip_stored(QuantValue::E5M2, QuantStore::Native, &data);

    assert_eq!(
        computed, data,
        "an e5m2 code point did not reconstruct as itself in the native store"
    );
}

/// The two stores are two spellings of one format, so they must agree value for value. A packed
/// path encoding in software and a native one converting on the backend could each be defensible
/// alone while disagreeing with each other, which would make a tensor's reconstruction depend on
/// how it happened to be stored.
#[test]
fn the_native_and_packed_stores_agree() {
    if !native_store_is_addressable() {
        return;
    }

    // Values off the grid on purpose: where the two could differ is in how they round, so asking
    // only about code points would not tell them apart.
    let data = [0.3f32, 1.7, -2.2, 0.09, 300.0, -300.0, 17.0, -0.001];

    for value in [QuantValue::E4M3, QuantValue::E5M2] {
        let packed = round_trip_stored(value, QuantStore::PackedU32(0), &data);
        let native = round_trip_stored(value, QuantStore::Native, &data);

        assert_eq!(
            packed, native,
            "{value:?} reconstructs differently depending on its store"
        );
    }
}
