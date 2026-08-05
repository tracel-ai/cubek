//! Block scales are rounded up to what their storage precision holds, never to the nearest value.
//!
//! Rounding to nearest can store a scale below the one the caller divided by, which puts that
//! block's extreme values past the quantization range, where they clamp. Rounding up keeps the
//! stored scale at or above the requested one, so the range still covers the block.

use cubecl::{
    features::TypeUsage,
    prelude::*,
    {TestRuntime, zspace::shape},
};
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

use super::harness::{f32_tensor, quantize, scale_shape};

const M: usize = 8;
const N: usize = 32;
const BLOCK: usize = 32;

/// The f16 mantissa step at exponent 0, the spacing of the grid a scale near 1.0 is stored on.
const F16_STEP: f32 = 1.0 / 1024.0;

/// Requested block scale paired with what f16 storage must hold for it. 1.0012 sits between the
/// grid points at one and two steps above 1.0, nearer the lower one, so rounding to nearest and
/// rounding up disagree: that disagreement is what this test pins down.
const SCALES: [(f32, f32); 4] = [
    (1.0, 1.0),
    (1.0012, 1.0 + 2.0 * F16_STEP),
    (1.5, 1.5),
    (1.0012, 1.0 + 2.0 * F16_STEP),
];

#[test]
fn block_scales_are_stored_rounded_up_to_their_storage_precision() {
    let client = TestRuntime::client(&Default::default());
    if !half::f16::supported_uses(&client).contains(TypeUsage::Conversion) {
        println!("f16 unsupported on this runtime, nothing checked");
        return;
    }

    let shape = shape![M, N];
    let num_blocks = M * N / BLOCK;

    let requested: Vec<f32> = (0..num_blocks)
        .map(|b| SCALES[b % SCALES.len()].0)
        .collect();
    let data: Vec<f32> = (0..M * N)
        .map(|i| ((i % 9) as f32 - 4.0) * requested[i / BLOCK])
        .collect();

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([BLOCK as u8]))
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F16)
        .with_mode(QuantMode::Symmetric);

    let input = f32_tensor(&client, &data, shape.clone());
    let scale = f32_tensor(&client, &requested, scale_shape(&scheme, &shape));

    // Only the quantize direction matters here: what is under test is the scale the kernel stored,
    // not what reading it back reconstructs.
    let (_, stored, _) = quantize(&client, &scheme, &input, &scale, None, &shape);

    let stored = client.read_one_unchecked_tensor(stored.into_copy_descriptor());
    let stored = half::f16::from_bytes(&stored);

    assert_eq!(stored.len(), num_blocks);
    for (block, &stored) in stored.iter().enumerate() {
        let (asked, expected) = SCALES[block % SCALES.len()];
        let stored = stored.to_f32();
        assert_eq!(
            stored, expected,
            "block {block} asked for {asked} and got {stored} stored, expected {expected}"
        );
        assert!(
            stored >= asked,
            "block {block} stored {stored} below the requested {asked}, so its extreme values \
             quantize past the range and clamp"
        );
    }
}
