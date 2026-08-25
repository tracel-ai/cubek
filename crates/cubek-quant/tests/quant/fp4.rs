//! `e2m1`: that a packed fp4 tensor reconstructs on the *format's* grid rather than on the
//! integers of the same width.
//!
//! The round trip over the integer value types lives in `symmetric.rs`, and this is exactly what
//! that file cannot check. An integer format's codes and its magnitudes are the same numbers, so a
//! codec that confuses the two round-trips every value there and looks correct. `e2m1` is where
//! the two part — its eight magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, so a codec that reads
//! the field as a small signed integer silently drops `0.5` and `1.5`, gains a `5` the format
//! cannot represent, and leaves an `int4` clamped to ±6 wearing fp4's name. That degradation is
//! what these tests exist to catch: it raises no error and costs accuracy on every weight.

use cubecl::{
    client::ComputeClient,
    features::TypeUsage,
    prelude::*,
    {TestRuntime, zspace::shape},
};
use cubek_quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};

use super::harness::{dequantize, f32_tensor, quantize};

/// The `e2m1` magnitudes, in code order. Written out rather than derived: a table derived from the
/// codec under test would agree with it however wrong it is.
const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// One block, holding every code the format has: the eight magnitudes and their negatives.
const BLOCK: usize = 16;

fn scheme(two_level: bool) -> QuantScheme {
    let scheme = QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .with_store(QuantStore::PackedU32(0))
        .with_mode(QuantMode::Symmetric)
        .per_block([BLOCK as u8], ScaleDtype::F32);

    match two_level {
        // NVFP4's own shape, and the reason the format is interesting: `ue4m3` block scales are
        // only affordable under a per-tensor factor.
        true => scheme.per_tensor(ScaleDtype::F32),
        false => scheme,
    }
}

/// Every code the format has, once, in code order and then negated.
fn every_code() -> Vec<f32> {
    MAGNITUDES
        .iter()
        .copied()
        .chain(MAGNITUDES.iter().map(|magnitude| -magnitude))
        .collect()
}

/// Round trip `data` at an exact scale of one, so the reconstruction is the codec's answer and
/// nothing else. `global` rides along for the two-level scheme, which needs the binding.
fn round_trip(scheme: &QuantScheme, data: &[f32]) -> Vec<f32> {
    let client = TestRuntime::client(&Default::default());
    let shape = shape![1, BLOCK];

    let input = f32_tensor(&client, data, shape.clone());
    let scale = f32_tensor(&client, &[1.0], shape![1, 1]);
    // One, so the level composes without moving anything: a two-level scheme that lost the factor
    // would still be caught by `two_level.rs`, and here it would only blur what is being asked.
    let global = scheme
        .tensor_scale()
        .filter(|_| scheme.block_scale().is_some())
        .map(|_| f32_tensor(&client, &[1.0], shape![1]));

    let (values, scales, out_global) =
        quantize(&client, scheme, &input, &scale, global.as_ref(), &shape);
    let out = dequantize(
        &client,
        scheme,
        &values,
        &scales,
        out_global.as_ref(),
        &shape,
        f32::elem_type_native(),
    );

    let computed = client.read_one_unchecked_tensor(out.into_copy_descriptor());
    f32::from_bytes(&computed).to_vec()
}

/// At a scale of one, every code point is already representable, so the round trip is the
/// identity. This is the test the integer codec failed: it returned `1.0` for `0.5` and `2.0` for
/// `1.5`, having rounded both to whole numbers before ever reaching the format.
#[test]
fn every_code_point_survives_the_round_trip() {
    let data = every_code();
    let computed = round_trip(&scheme(false), &data);

    assert_eq!(
        computed, data,
        "an e2m1 code point did not reconstruct as itself"
    );
}

/// The same, under the per-tensor level NVFP4 carries — the composition must not disturb the grid.
#[test]
fn every_code_point_survives_a_two_level_round_trip() {
    let data = every_code();
    let computed = round_trip(&scheme(true), &data);

    assert_eq!(
        computed, data,
        "an e2m1 code point did not reconstruct as itself under a per-tensor scale"
    );
}

/// Whatever goes in, what comes out is a code point. An integer codec fails this on `5.0`, which
/// it reproduces exactly and `e2m1` cannot represent at all.
#[test]
fn everything_reconstructs_onto_the_grid() {
    // A sweep across the range at a step that shares no factor with the grid, so values land
    // between code points rather than on them.
    let data: Vec<f32> = (0..BLOCK).map(|i| -6.0 + i as f32 * 0.7).collect();
    let computed = round_trip(&scheme(false), &data);

    for value in computed {
        assert!(
            MAGNITUDES.contains(&value.abs()),
            "{value} is not an e2m1 code point"
        );
    }
}

/// Values between code points take the nearer one, and an exact midpoint takes the even code.
///
/// Ties are not an edge case on a grid this coarse — `0.75` and `2.5` are both exact midpoints of
/// neighbouring code points, and a codec that rounded every one of them outward would bias each
/// quantized block upward by a visible amount rather than by a rounding error.
#[test]
fn values_between_code_points_round_to_nearest_ties_to_even() {
    let cases = [
        // Nearer neighbour.
        (0.2, 0.0),
        (0.3, 0.5),
        (1.4, 1.5),
        (2.6, 3.0),
        (4.9, 4.0),
        (5.5, 6.0),
        // Exact midpoints, which go to the even code.
        (0.25, 0.0),
        (0.75, 1.0),
        (1.25, 1.0),
        (1.75, 2.0),
        (2.5, 2.0),
        (3.5, 4.0),
        (5.0, 4.0),
        // Past the top magnitude, which saturates: e2m1 has no infinity to reach.
        (7.0, 6.0),
        (100.0, 6.0),
        (0.0, 0.0),
    ];

    let data: Vec<f32> = cases.iter().map(|(input, _)| *input).collect();
    let computed = round_trip(&scheme(false), &data);

    for ((input, expected), got) in cases.iter().zip(computed) {
        assert_eq!(
            got, *expected,
            "{input} quantized to {got}, wanted {expected}"
        );
    }
}

/// The negative half of the grid is the positive half's mirror, sign included. A codec deriving
/// the sign from a two's-complement reading of the field gets this wrong in a way the magnitudes
/// alone would not show.
#[test]
fn the_sign_is_carried() {
    let data: Vec<f32> = (0..BLOCK)
        .map(|i| {
            let magnitude = MAGNITUDES[i % MAGNITUDES.len()];
            if i % 2 == 0 { magnitude } else { -magnitude }
        })
        .collect();
    let computed = round_trip(&scheme(false), &data);

    assert_eq!(computed, data, "an e2m1 sign was lost in the round trip");
}

/// MXFP4, the OCP microscaling format: the same `e2m1` values in blocks of 32, scaled by one
/// power-of-two `ue8m0` exponent per block and no per-tensor level.
///
/// The scale being a bare exponent is what makes these tests sharper than the ones above: a power
/// of two is exact in f32, so a correct round trip reproduces its input *exactly* rather than
/// within a scale-rounding tolerance. Anything approximate here is a real defect.
mod mx {
    use super::*;

    /// MXFP4's block, as the format fixes it.
    const MX_BLOCK: usize = 32;

    fn mxfp4() -> QuantScheme {
        QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedU32(0))
            .with_mode(QuantMode::Symmetric)
            .per_block([MX_BLOCK as u8], ScaleDtype::UE8M0)
    }

    /// Whether this runtime can hold a `ue8m0` scale at all.
    ///
    /// The scale is read one at a time, and an 8-bit float is a byte on every backend that has
    /// one — the software codec converts it, but something has to *address* it. WGSL has no 8-bit
    /// type, so it packs fp8 four lanes to a `u32` and a scalar has no representation there; the
    /// `ue4m3` test in `two_level.rs` sits out the same backends for the same reason.
    fn addressable(client: &ComputeClient<TestRuntime>) -> bool {
        if u8::supported_uses(client).contains(TypeUsage::Conversion) {
            return true;
        }
        println!("no 8-bit type on this runtime, a ue8m0 scale has nowhere to live");
        false
    }

    /// Round trip one block at `scale`, which the codec stores as `ue8m0` and may round up.
    /// `None` where the runtime cannot address the scale.
    fn round_trip_at(scale: f32, data: &[f32]) -> Option<Vec<f32>> {
        let client = TestRuntime::client(&Default::default());
        if !addressable(&client) {
            return None;
        }
        let shape = shape![1, MX_BLOCK];
        let scheme = mxfp4();

        let input = f32_tensor(&client, data, shape.clone());
        let scales = f32_tensor(&client, &[scale], shape![1, 1]);

        let (values, out_scales, _) = quantize(&client, &scheme, &input, &scales, None, &shape);
        let out = dequantize(
            &client,
            &scheme,
            &values,
            &out_scales,
            None,
            &shape,
            f32::elem_type_native(),
        );

        let computed = client.read_one_unchecked_tensor(out.into_copy_descriptor());
        Some(f32::from_bytes(&computed).to_vec())
    }

    /// Every code point, scaled by a power of two, comes back bit-exact. Both factors are exact in
    /// f32, so there is nothing for a tolerance to hide.
    #[test]
    fn a_power_of_two_scale_round_trips_exactly() {
        for exp in [-8i32, -1, 0, 1, 8] {
            let scale = 2f32.powi(exp);
            let data: Vec<f32> = (0..MX_BLOCK)
                .map(|i| {
                    let magnitude = MAGNITUDES[(i / 2) % MAGNITUDES.len()] * scale;
                    if i % 2 == 0 { magnitude } else { -magnitude }
                })
                .collect();

            let Some(computed) = round_trip_at(scale, &data) else {
                return;
            };
            assert_eq!(
                computed, data,
                "2^{exp} scaled block did not round trip exactly"
            );
        }
    }

    /// A calibration that is not a power of two is stored as the next one **up**. Up is what keeps
    /// the block's largest value inside the quantization range; rounding down would clip it.
    #[test]
    fn a_scale_between_powers_of_two_rounds_up() {
        // 1.5 is stored as 2, so a value at 6 * 1.5 reconstructs against a scale of 2 and lands on
        // the nearest code point to 4.5 — which is 4, not 4.5, since e2m1 has no 4.5.
        let data: Vec<f32> = core::iter::repeat_n(9.0f32, MX_BLOCK).collect();
        let Some(computed) = round_trip_at(1.5, &data) else {
            return;
        };

        for value in computed {
            assert_eq!(value, 8.0, "9.0 under a scale of 1.5 landed at {value}");
        }
    }

    /// A block of zeros survives. `ue8m0` has no code for a zero scale, so the codec has to clamp
    /// it to the format's minimum rather than store it — and a scale of 2^-127 still reconstructs
    /// zeros as zeros.
    #[test]
    fn an_all_zero_block_survives_a_scale_it_cannot_store() {
        let data: Vec<f32> = core::iter::repeat_n(0.0f32, MX_BLOCK).collect();

        let Some(computed) = round_trip_at(0.0, &data) else {
            return;
        };
        assert_eq!(computed, data, "a zero-scaled block did not come back zero");
    }

    /// A scale past `ue8m0`'s top pins at 2^127 instead of wrapping into a different exponent,
    /// which is what an unclamped step on the bit pattern would do.
    ///
    /// Checked as an agreement rather than against an expected number: every scale at or above the
    /// maximum stores the same code, so they all have to reconstruct identically. A wrap would
    /// send them to unrelated exponents and they would not.
    #[test]
    fn a_scale_past_the_top_saturates_rather_than_wrapping() {
        let data: Vec<f32> = core::iter::repeat_n(1.0f32, MX_BLOCK).collect();

        let Some(at_max) = round_trip_at(ScaleDtype::UE8M0_MAX, &data) else {
            return;
        };
        for scale in [ScaleDtype::UE8M0_MAX * 1.5, f32::MAX] {
            assert_eq!(
                round_trip_at(scale, &data).unwrap(),
                at_max,
                "{scale:e} did not store the same scale as the maximum"
            );
        }
        // And nothing reconstructs to a non-finite value on the way. At this scale the block's
        // values are 38 orders of magnitude below one quantization step, so they *do* land on
        // zero — that is the format working, not the clamp failing.
        assert!(at_max.iter().all(|value| value.is_finite()));
    }
}
