use cubecl::quant::scheme::QuantMode;
use cubecl_common::{
    e4m3, e5m2,
    quant::scheme::{QuantScheme, QuantStore, QuantValue},
};

pub fn quantize(
    values: &[f32],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scheme: &QuantScheme,
) -> Vec<u8> {
    let scales_shape = scales_shape(shape, block_dims);
    let quantized = quantized_values(values, shape, scales, block_dims, &scales_shape, scheme);

    match scheme.store {
        QuantStore::Native => encode_native(&quantized, scheme.value),
        QuantStore::PackedU32(_) => encode_packed_u32(&quantized, scheme),
        other => panic!("quantize stub: unsupported store {other:?}"),
    }
}

#[allow(dead_code)]
pub fn dequantize(
    bytes: &[u8],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scheme: &QuantScheme,
) -> Vec<f32> {
    let scales_shape = scales_shape(shape, block_dims);

    let quants = match scheme.store {
        QuantStore::Native => decode_native(bytes, scheme.value),
        QuantStore::PackedU32(_) => decode_packed_u32(bytes, scheme),
        other => panic!("dequantize stub: unsupported store {other:?}"),
    };

    dequantized_values(&quants, scales, shape, block_dims, &scales_shape, scheme)
}

fn quantized_values(
    values: &[f32],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scales_shape: &[usize],
    scheme: &QuantScheme,
) -> Vec<f32> {
    let (range_min, range_max) = scheme.value.range();

    match scheme.mode {
        QuantMode::Symmetric => values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let scale = scales[scale_index(i, shape, block_dims, scales_shape)];
                (v / scale).round().clamp(range_min, range_max)
            })
            .collect(),
    }
}

#[allow(dead_code)]
fn dequantized_values(
    quantized: &[f32],
    scales: &[f32],
    shape: &[usize],
    block_dims: &[usize],
    scales_shape: &[usize],
    scheme: &QuantScheme,
) -> Vec<f32> {
    match scheme.mode {
        QuantMode::Symmetric => quantized
            .iter()
            .enumerate()
            .map(|(i, &q)| q * scales[scale_index(i, shape, block_dims, scales_shape)])
            .collect(),
    }
}

/// 1:1 native storage: `i8` for `Q8*`, fp8 bits for `E4M3`/`E5M2`.
fn encode_native(quantized: &[f32], value: QuantValue) -> Vec<u8> {
    match value {
        QuantValue::Q8F | QuantValue::Q8S => quantized.iter().map(|&q| q as i8 as u8).collect(),
        QuantValue::E4M3 => quantized
            .iter()
            .map(|&q| e4m3::from_f32(q).to_bits())
            .collect(),
        QuantValue::E5M2 => quantized
            .iter()
            .map(|&q| e5m2::from_f32(q).to_bits())
            .collect(),
        other => panic!("quantize stub: {other:?} is not supported for native storage"),
    }
}

#[allow(dead_code)]
fn decode_native(bytes: &[u8], value: QuantValue) -> Vec<f32> {
    match value {
        QuantValue::Q8F | QuantValue::Q8S => bytes.iter().map(|&b| b as i8 as f32).collect(),
        QuantValue::E4M3 => bytes.iter().map(|&b| e4m3::from_bits(b).to_f32()).collect(),
        QuantValue::E5M2 => bytes.iter().map(|&b| e5m2::from_bits(b).to_f32()).collect(),
        other => panic!("dequantize stub: {other:?} is not supported for native storage"),
    }
}

/// Pack `num_quants` consecutive quants (along the innermost dimension) into
/// each `u32`, low bits first, matching `pack_q`.
fn encode_packed_u32(quantized: &[f32], scheme: &QuantScheme) -> Vec<u8> {
    let size_quant = scheme.size_bits_value();
    let num_quants = scheme.num_quants();
    let mask = quant_mask(size_quant);

    let packed: Vec<u32> = quantized
        .chunks(num_quants)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u32, |acc, (p, &q)| {
                let bits = (q as i32 as u32) & mask;
                acc | (bits << (p * size_quant))
            })
        })
        .collect();

    bytemuck::cast_slice(&packed).to_vec()
}

/// Inverse of [`encode_packed_u32`]: unpack each `u32` into `num_quants`
/// sign-extended quant values, matching `unpack_q`.
#[allow(dead_code)]
fn decode_packed_u32(bytes: &[u8], scheme: &QuantScheme) -> Vec<f32> {
    let size_quant = scheme.size_bits_value();
    let num_quants = scheme.num_quants();
    let mask = quant_mask(size_quant);
    let sign_bit = 1u32 << (size_quant - 1);

    let mut out = Vec::with_capacity((bytes.len() / 4) * num_quants);
    for &word in bytemuck::cast_slice::<u8, u32>(bytes) {
        for p in 0..num_quants {
            let raw = (word >> (p * size_quant)) & mask;
            // Two's-complement sign extension.
            let q = if raw & sign_bit != 0 {
                (raw | !mask) as i32
            } else {
                raw as i32
            };
            out.push(q as f32);
        }
    }
    out
}

fn quant_mask(size_quant: usize) -> u32 {
    if size_quant >= 32 {
        u32::MAX
    } else {
        (1u32 << size_quant) - 1
    }
}

/// Shape of the per-block scale grid: each dimension divided by its block
/// extent.
fn scales_shape(shape: &[usize], block_dims: &[usize]) -> Vec<usize> {
    assert_eq!(
        shape.len(),
        block_dims.len(),
        "shape/block_dims rank mismatch"
    );
    shape
        .iter()
        .zip(block_dims)
        .map(|(&d, &b)| {
            assert!(
                d.is_multiple_of(b),
                "block dim {b} must divide dimension {d}"
            );
            d / b
        })
        .collect()
}

/// Map a logical (row-major) element index to the index of its block in the
/// row-major scales grid.
fn scale_index(
    linear: usize,
    shape: &[usize],
    block_dims: &[usize],
    scales_shape: &[usize],
) -> usize {
    let rank = shape.len();
    let mut rem = linear;
    let mut coord = vec![0usize; rank];
    for d in (0..rank).rev() {
        coord[d] = rem % shape[d];
        rem /= shape[d];
    }

    let mut block_linear = 0;
    for d in 0..rank {
        block_linear = block_linear * scales_shape[d] + coord[d] / block_dims[d];
    }
    block_linear
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantStore, QuantValue};

    fn scheme(value: QuantValue, store: QuantStore, level: QuantLevel) -> QuantScheme {
        QuantScheme::default()
            .with_mode(QuantMode::Symmetric)
            .with_level(level)
            .with_value(value)
            .with_store(store)
            .with_param(QuantParam::F32)
    }

    /// Quantize then dequantize and assert the round-trip stays within one
    /// quantization step (`scale`), the worst-case symmetric error.
    fn assert_round_trips(value: QuantValue, store: QuantStore, scale: f32) {
        let s = scheme(value, store, QuantLevel::Tensor);
        let n = 64; // multiple of every num_quants we test (≤16)
        let values: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();

        let bytes = quantize(&values, &[n], &[scale], &[n], &s);
        let restored = dequantize(&bytes, &[n], &[scale], &[n], &s);

        let max_err = scale + 1e-6;
        for (got, want) in restored.iter().zip(&values) {
            assert!(
                (got - want).abs() <= max_err,
                "{value:?}/{store:?}: dequant {got} too far from {want}",
            );
        }
    }

    #[test]
    fn native_q8_is_one_to_one_i8() {
        let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let scale = 1.0 / 127.0;
        let bytes = quantize(
            &values,
            &[values.len()],
            &[scale],
            &[values.len()],
            &scheme(QuantValue::Q8S, QuantStore::Native, QuantLevel::Tensor),
        );

        let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(got, vec![-127, -64, 0, 64, 127]);
    }

    #[test]
    fn round_trips_for_integer_schemes() {
        assert_round_trips(QuantValue::Q8S, QuantStore::Native, 1.0 / 127.0);
        assert_round_trips(QuantValue::Q8S, QuantStore::PackedU32(0), 1.0 / 127.0);
        assert_round_trips(QuantValue::Q4S, QuantStore::PackedU32(0), 1.0 / 7.0);
        assert_round_trips(QuantValue::Q2S, QuantStore::PackedU32(0), 1.0 / 1.0);
    }

    #[test]
    fn round_trips_for_fp8_native() {
        // Like the kernel, fp8 quantization rounds to an integer and then casts
        // to fp8 — so the stored value also carries fp8 representation error.
        // Pick scales that keep the quants within each format's exactly
        // representable integer range (E4M3: |q| ≤ 8, E5M2: |q| ≤ 4).
        assert_round_trips(QuantValue::E4M3, QuantStore::Native, 1.0 / 8.0);
        assert_round_trips(QuantValue::E5M2, QuantStore::Native, 1.0 / 4.0);
    }

    #[test]
    fn block_level_uses_per_block_scale() {
        // Two blocks of 4 along the last dim, each with its own scale.
        let s = scheme(
            QuantValue::Q8S,
            QuantStore::Native,
            QuantLevel::block([4u8]),
        );
        let values = vec![
            0.1, 0.2, 0.3, 0.4, // block 0
            10.0, 20.0, 30.0, 40.0, // block 1
        ];
        let scales = vec![0.4 / 127.0, 40.0 / 127.0];

        let bytes = quantize(&values, &[8], &scales, &[4], &s);
        let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();

        // Largest element of each block saturates to the quant max.
        assert_eq!(got[3], 127);
        assert_eq!(got[7], 127);

        // And the per-block scale is applied on the way back.
        let restored = dequantize(&bytes, &[8], &scales, &[4], &s);
        for (got, want) in restored.iter().zip(&values) {
            let step = if *want < 1.0 { scales[0] } else { scales[1] };
            assert!((got - want).abs() <= step + 1e-6);
        }
    }
}
