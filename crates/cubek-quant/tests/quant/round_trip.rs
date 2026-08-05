// Generated once per value type and shape by `testgen_quant!`, which is why this file declares no
// imports of its own. See `mod.rs`.

use cubecl::{
    features::TypeUsage,
    {TestRuntime, zspace::shape},
};
use cubek_quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue};

/// Room for the scale itself travelling through its storage type, on top of the half step the
/// quantization costs.
const REL_TOL: f32 = 1e-4;

#[test]
fn test_quantization_symmetric_tensor() {
    tensor_round_trip(SHAPE_X, SHAPE_Y, VALUE);
}

#[test]
fn test_quantization_symmetric_block() {
    // Shape x as the block size, so one row is one block.
    block_round_trip(
        SHAPE_X,
        SHAPE_Y,
        VALUE,
        SHAPE_X,
        QuantStore::PackedU32(0),
        false,
    );
}

#[test]
fn test_quantization_symmetric_block_tensor() {
    // Native storage holds byte-wide values only; the packed sibling covers every value.
    if !matches!(VALUE, QuantValue::Q8F | QuantValue::Q8S) {
        return;
    }

    block_round_trip(SHAPE_X, SHAPE_Y, VALUE, SHAPE_X, QuantStore::Native, true);
}

#[test]
fn test_quantization_symmetric_block_tensor_packed() {
    block_round_trip(
        SHAPE_X,
        SHAPE_Y,
        VALUE,
        SHAPE_X,
        QuantStore::PackedU32(0),
        true,
    );
}

/// One scale for the whole tensor, so the same half step bounds the error everywhere.
fn tensor_round_trip(m: usize, n: usize, value: QuantValue) {
    let client = TestRuntime::client(&Default::default());
    let shape = shape![m, n];

    let num_elems = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<f32> = (0..num_elems).map(|v| v as f32 - half).collect();

    // The input data range is not affected by quant range symmetry.
    let (q_min, q_max) = value.range();
    let scale_f32 = (2.0 * half) / (q_max - q_min);

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    let input = f32_tensor(&client, &data, shape.clone());
    let scale = f32_tensor(&client, &[scale_f32], scale_shape(&scheme, &shape));

    let (values, scales) = quantize(&client, &scheme, &input, &scale, None, &shape);
    let out = dequantize(
        &client,
        &scheme,
        &values,
        &scales,
        None,
        &shape,
        f32::as_type_native_unchecked().storage_type(),
    );

    let computed = client.read_one_unchecked_tensor(out.into_copy_descriptor());
    let restored = f32::from_bytes(&computed);

    // Max quantization error = step size / 2.
    let max_error = (scale_f32 / 2.0) * (1.0 + REL_TOL);
    assert_eq!(restored.len(), data.len());
    for (actual, expected) in restored.iter().zip(data) {
        let diff = f32::abs(actual - expected);
        assert!(
            diff <= max_error,
            "Expected: {expected} | Actual: {actual} (diff {diff} > {max_error})"
        );
    }
}

/// A scale per block, and with `two_level` a global scale normalizing them.
///
/// The block scales are then deliberately split, so that neither level alone reconstructs the
/// data and dropping the global scale or applying it twice cannot hide inside the tolerance.
fn block_round_trip(
    m: usize,
    n: usize,
    value: QuantValue,
    block_size: usize,
    store: QuantStore,
    two_level: bool,
) {
    let client = TestRuntime::client(&Default::default());
    if store == QuantStore::Native && !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu), which packed storage does not need
    }

    let shape = shape![m, n];

    let num_elems = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<f32> = (0..num_elems)
        .map(|v| (v as f32 - half) / num_elems as f32)
        .collect();

    // Symmetric quantization assumes a zero bias, so a block's range is twice its largest
    // magnitude.
    let (q_min, q_max) = value.range();
    let calibrated: Vec<f32> = data
        .chunks(block_size)
        .map(|block| {
            let amax = block.iter().fold(0.0f32, |amax, v| amax.max(v.abs()));
            2.0 * amax / (q_max - q_min)
        })
        .collect();

    // Under a global scale the block scales carry only what is left of the spread, so the largest
    // of them lands at a quarter of what it would be on its own.
    let global_f32 = two_level.then(|| calibrated.iter().copied().fold(0.0f32, f32::max) / 4.0);
    let scales: Vec<f32> = match global_f32 {
        Some(global) => calibrated.iter().map(|scale| scale / global).collect(),
        None => calibrated,
    };

    let level = match global_f32 {
        Some(_) => QuantLevel::block_tensor([block_size as u8], QuantParam::F32),
        None => QuantLevel::block([block_size as u8]),
    };
    let scheme = QuantScheme::default()
        .with_level(level)
        .with_value(value)
        .with_store(store)
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    let input = f32_tensor(&client, &data, shape.clone());
    let scale = f32_tensor(&client, &scales, scale_shape(&scheme, &shape));
    let global = global_f32.map(|global| f32_tensor(&client, &[global], shape![1]));

    let (values, stored) = quantize(&client, &scheme, &input, &scale, global.as_ref(), &shape);
    let out = dequantize(
        &client,
        &scheme,
        &values,
        &stored,
        global.as_ref(),
        &shape,
        f32::as_type_native_unchecked().storage_type(),
    );

    if let (Some(handle), Some(expected)) = (&global, global_f32) {
        // Quantization reads the global scale and leaves it where it found it.
        let written = client.read_one_unchecked_tensor(handle.clone().into_copy_descriptor());
        assert_eq!(f32::from_bytes(&written)[0], expected);
    }

    let computed = client.read_one_unchecked_tensor(out.into_copy_descriptor());
    let restored = f32::from_bytes(&computed);

    assert_eq!(restored.len(), data.len());
    for (block, chunk) in data.chunks(block_size).enumerate() {
        // Half a step of the effective scale, which under two levels is the product of both.
        let effective = global_f32.unwrap_or(1.0) * scales[block];
        let max_error = (effective / 2.0) * (1.0 + REL_TOL);

        for (i, expected) in chunk.iter().enumerate() {
            let actual = restored[block * block_size + i];
            let diff = f32::abs(actual - expected);
            assert!(
                diff <= max_error,
                "block {block} index {i}: got {actual}, expected {expected}, \
                 diff {diff} over {max_error}"
            );
        }
    }
}
