//! `e4m3` and `e5m2` under `QuantStore::PackedU32`: that a packed fp8 tensor reconstructs on the
//! *format's* grid rather than on the integers of the same width.
//!
//! The same question `fp4.rs` asks, and it belongs here for the same reason: a packed field holds
//! the format's *code*, and only for the integer value types is a code the same number as the
//! magnitude it stands for. `e2m1` is where that first bites, but `e4m3` is where it bites hardest
//! — its codes run to ±448 while a byte read as two's complement stops at ±127, so a codec that
//! confuses the two does not merely lose precision, it wraps. `300.0` came back as `44.0`.
//!
//! The native store has its own coverage; these are the packed path, which has no fp8 type in play
//! at all and so has to encode in software.

use cubecl::{prelude::*, {TestRuntime, zspace::shape}};
use cubek_quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};

use super::harness::{dequantize, f32_tensor, quantize};

/// One block, sized to the vector the packed path builds.
const BLOCK: usize = 8;

fn scheme(value: QuantValue) -> QuantScheme {
    QuantScheme::default()
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_mode(QuantMode::Symmetric)
        .per_block([BLOCK as u8], ScaleDtype::F32)
}

/// Round trip `data` at an exact scale of one, so the reconstruction is the codec's answer and
/// nothing else.
fn round_trip(value: QuantValue, data: &[f32]) -> Vec<f32> {
    let client = TestRuntime::client(&Default::default());
    let scheme = scheme(value);
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

/// Magnitudes above what a byte read as two's complement can hold. This is the regression the
/// packed path actually had: the field was cast from the value rather than encoded, so anything
/// past ±127 wrapped out of the byte into an unrelated magnitude.
#[test]
fn large_magnitudes_do_not_wrap_out_of_the_field() {
    // 300 is not an e4m3 value; 288 and 320 are its neighbours and 304 the midpoint, so it rounds
    // down to 288. What matters is that it lands near itself rather than on 44.
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
