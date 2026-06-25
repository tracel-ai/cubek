//! Host-side stand-in for `cubek_quant::quantize::launch_ref`.
//!
//! Reimplements just the symmetric integer quantization the test utilities
//! exercise so `cubek-test-utils` doesn't have to depend on `cubek-quant`:
//!
//! * values  — `Q8`/`Q4`/`Q2`, full or symmetric range (the integer schemes);
//! * stores  — [`QuantStore::Native`] (1:1 `i8`) and [`QuantStore::PackedU32`]
//!   (several quants bit-packed into a `u32`, innermost dimension);
//! * levels  — [`QuantLevel::Tensor`] and [`QuantLevel::Block`].
//!
//! The companion `out_scale` buffer the real kernel writes is simply the input
//! scales cast to the param precision, so it isn't produced here — the caller
//! already has the scales it passed in.

use cubecl::quant::scheme::QuantMode;
use cubecl_common::quant::scheme::{QuantScheme, QuantStore};

/// Quantize `values` into the packed/native output buffer bytes, mirroring
/// `cubek_quant::quantize::launch_ref` for the schemes listed in the module
/// docs.
///
/// * `values` — logical, row-major input data.
/// * `shape` — logical shape of `values`.
/// * `scales` — one scale per block, row-major over the block grid (a single
///   element for [`QuantLevel::Tensor`]).
/// * `block_dims` — per-dimension block extent (the full dimension for
///   tensor-level quantization). Must divide `shape` element-wise.
pub fn quantize(
    values: &[f32],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scheme: &QuantScheme,
) -> Vec<u8> {
    assert_eq!(
        shape.len(),
        block_dims.len(),
        "shape/block_dims rank mismatch"
    );

    let scales_shape: Vec<usize> = shape
        .iter()
        .zip(block_dims)
        .map(|(&d, &b)| {
            assert!(
                d.is_multiple_of(b),
                "block dim {b} must divide dimension {d}"
            );
            d / b
        })
        .collect();

    let (range_min, range_max) = scheme.value.range();

    let quantized: Vec<f32> = match scheme.mode {
        QuantMode::Symmetric => values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let scale = scales[scale_index(i, shape, block_dims, &scales_shape)];
                (v / scale).round().clamp(range_min, range_max)
            })
            .collect(),
    };

    match scheme.store {
        // 1:1 native storage — only `Q8*` (i8) reaches this path.
        QuantStore::Native => quantized.iter().map(|&q| q as i8 as u8).collect(),
        // Pack `num_quants` consecutive quants (along the innermost dimension)
        // into each `u32`, low bits first, matching `pack_q`.
        QuantStore::PackedU32(_) => {
            let size_quant = scheme.size_bits_value();
            let num_quants = scheme.num_quants();
            let mask: u32 = if size_quant >= 32 {
                u32::MAX
            } else {
                (1u32 << size_quant) - 1
            };

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
        other => panic!("quantize stub: unsupported store {other:?}"),
    }
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

    /// Dequantize a `u32`-packed buffer back to `f32` so we can check the
    /// round-trip stays within the quantization step.
    fn dequant_packed(bytes: &[u8], scale: f32, scheme: &QuantScheme) -> Vec<f32> {
        let size_quant = scheme.size_bits_value();
        let num_quants = scheme.num_quants();
        let mask: u32 = (1u32 << size_quant) - 1;
        let sign_bit = 1u32 << (size_quant - 1);

        let mut out = Vec::new();
        for &word in bytemuck::cast_slice::<u8, u32>(bytes) {
            for p in 0..num_quants {
                let raw = (word >> (p * size_quant)) & mask;
                // Sign-extend the two's-complement quant value.
                let q = if raw & sign_bit != 0 {
                    (raw | !mask) as i32
                } else {
                    raw as i32
                };
                out.push(q as f32 * scale);
            }
        }
        out
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
    fn packed_u32_round_trips_within_step() {
        let s = scheme(
            QuantValue::Q4S,
            QuantStore::PackedU32(0),
            QuantLevel::Tensor,
        );
        let n = 64; // multiple of num_quants (8)
        let values: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();
        let scale = 1.0 / 7.0;

        let bytes = quantize(&values, &[n], &[scale], &[n], &s);
        let restored = dequant_packed(&bytes, scale, &s);

        let max_err = scale / 2.0 + 1e-6;
        for (got, want) in restored.iter().zip(&values) {
            assert!(
                (got - want).abs() <= max_err,
                "dequant {got} too far from {want}",
            );
        }
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
    }
}
