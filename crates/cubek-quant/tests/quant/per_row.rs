//! FP8-row, and MXFP4 with row-factored scales: that each launch writes the levels its scheme
//! names, and that the values they reconstruct are the input's, to the format's rounding.

use cubecl::{
    client::Client,
    ir::{ElemType, UIntKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_quant::per_row::{
    EXACT_LOWEST_OFFSET, MXFP4_BLOCK, factor_mxfp4_scales_by_row, quantize_fp8_per_row,
    quantize_mxfp4_row_factored,
};

use super::harness::f32_tensor;

fn e2m1(code: u32) -> f32 {
    cubecl_common::e2m1::from_bits(code as u8).to_f32()
}

fn e4m3(byte: u8) -> f32 {
    cubecl_common::e4m3::from_bits(byte).to_f32()
}

fn ue8m0(byte: u8) -> f32 {
    2f32.powi(byte as i32 - 127)
}

fn read_bytes(client: &Client, handle: &TensorHandle) -> Vec<u8> {
    client
        .read_one_unchecked_tensor(handle.clone().into_copy_descriptor())
        .to_vec()
}

fn read_f32(client: &Client, handle: &TensorHandle) -> Vec<f32> {
    f32::from_bytes(&read_bytes(client, handle)).to_vec()
}

fn read_u32(client: &Client, handle: &TensorHandle) -> Vec<u32> {
    u32::from_bytes(&read_bytes(client, handle)).to_vec()
}

/// A row's values spread over a wide range: its own magnitude, a largest of `largest`, the rest
/// falling away from it.
fn row(len: usize, largest: f32, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let fall = ((i * 7 + seed * 13) % 23) as f32 / 23.0;
            let sign = if (i + seed).is_multiple_of(3) {
                -1.0
            } else {
                1.0
            };
            match i == seed % len {
                true => largest,
                false => sign * largest * fall,
            }
        })
        .collect()
}

/// Each row under its largest magnitude over `448`, each value its nearest `e4m3` under it, a
/// zero row under a scale of one. Three rows of 36: the row's words are not a cube's multiple.
#[test]
fn fp8_per_row_scales_each_row_by_its_largest() {
    let client = cubecl::test_device().client();
    let (rows, k) = (3, 36);
    let mut data = row(k, 1000.0, 1);
    data.extend(row(k, 0.003, 2));
    data.extend(vec![0.0; k]);
    let input = f32_tensor(&client, &data, shape![rows, k]);

    let quantized = quantize_fp8_per_row(&client, &input, f32::elem_type_native());

    let scales = read_f32(&client, &quantized.scales);
    assert_eq!(scales, vec![1000.0 / 448.0, 0.003 / 448.0, 1.0]);
    let values = read_bytes(&client, &quantized.values);
    for (at, (&byte, &expected)) in values.iter().zip(&data).enumerate() {
        let scale = scales[at / k];
        let got = e4m3(byte) * scale;
        // Half an `e4m3` step, relative, or half its smallest subnormal under the scale.
        let tolerance = expected.abs() / 16.0 + scale * 2f32.powi(-10);
        assert!(
            (got - expected).abs() <= tolerance,
            "value {at}: {got} for {expected}"
        );
    }
}

/// The block scales of rows `rows`, each block's `ue8m0` byte, written as stored.
fn scale_bytes(client: &Client, rows: &[Vec<u8>]) -> TensorHandle {
    let blocks = rows[0].len();
    let bytes: Vec<u8> = rows.concat();
    TensorHandle::new_contiguous(
        [rows.len(), blocks],
        client.create_from_slice(&bytes),
        ElemType::UInt(UIntKind::U8),
    )
}

/// Each row's largest block scale becomes its `f32` scale and every block its offset below it;
/// the blocks more than [`EXACT_LOWEST_OFFSET`] below are counted, and one past a `ue8m0`'s range
/// flushes to the smallest offset.
#[test]
fn factoring_mxfp4_scales_by_row_stores_offsets_below_the_largest() {
    let client = cubecl::test_device().client();
    let rows = vec![
        vec![127, 125, 130, 128, 122, 123, 130, 130],
        vec![100, 100, 100, 100, 91, 92, 90, 100],
        vec![200, 2, 199, 200, 200, 200, 200, 200],
    ];
    let scales = scale_bytes(&client, &rows);

    let restated = factor_mxfp4_scales_by_row(&client, &scales);

    let row_scales = read_f32(&client, &restated.row_scales);
    let offsets = read_bytes(&client, &restated.block_scales);
    let lossy = read_u32(&client, &restated.lossy_blocks);
    let mut expected_lossy = 0;
    for (r, blocks) in rows.iter().enumerate() {
        let largest = *blocks.iter().max().unwrap();
        assert_eq!(row_scales[r], ue8m0(largest), "row {r}'s scale");
        for (b, &scale) in blocks.iter().enumerate() {
            let below = (largest - scale) as i32;
            let expected = (127 - below).max(0) as u8;
            assert_eq!(offsets[r * blocks.len() + b], expected, "row {r} block {b}");
            if below > -EXACT_LOWEST_OFFSET {
                expected_lossy += 1;
            }
            // Every block within range keeps its value: the two levels multiply back to it.
            if below <= 127 {
                assert_eq!(ue8m0(expected) * row_scales[r], ue8m0(scale));
            }
        }
    }
    assert_eq!(expected_lossy, 3, "rows 1 and 2 hold the lossy blocks");
    assert_eq!(lossy, vec![expected_lossy]);
}

/// A float weight straight to MXFP4 with row-factored scales, three rows of twelve blocks: each
/// block's codes the nearest `e2m1` under its scale, its largest landing on `6`, and its value
/// under the two levels the input's to `e2m1`'s rounding. A block of zeros stays zero, under the
/// smallest scale, which the lossy count takes for a block that rounds.
#[test]
fn quantize_mxfp4_row_factored_reconstructs_the_input() {
    let client = cubecl::test_device().client();
    let (rows, k) = (3, 384);
    let blocks = k / MXFP4_BLOCK;
    // Each row's blocks under `2^-3` to `2^2`, twice over: the largest five steps above the least.
    let largest_of = |block: usize| 6.0 * 2f32.powi((block % 6) as i32 - 3);
    let mut data = Vec::new();
    for r in 0..rows {
        for b in 0..blocks {
            match (r, b) {
                (1, 3) => data.extend(vec![0.0; MXFP4_BLOCK]),
                _ => data.extend(row(MXFP4_BLOCK, largest_of(b), b + r)),
            }
        }
    }
    let input = f32_tensor(&client, &data, shape![rows, k]);

    let quantized = quantize_mxfp4_row_factored(&client, &input, f32::elem_type_native());

    let codes = read_u32(&client, &quantized.codes);
    let offsets = read_bytes(&client, &quantized.scales.block_scales);
    let row_scales = read_f32(&client, &quantized.scales.row_scales);
    for (at, &expected) in data.iter().enumerate() {
        let (r, b) = (at / k, at / MXFP4_BLOCK);
        let code = (codes[at / 8] >> (4 * (at % 8))) & 0xF;
        let scale = ue8m0(offsets[b]) * row_scales[r];
        let got = e2m1(code) * scale;
        // Half the step between `e2m1`'s two largest codes, relative to the block's scale.
        assert!(
            (got - expected).abs() <= scale,
            "value {at}: {got} for {expected}"
        );
        if expected == largest_of(b % blocks) {
            assert_eq!(got, expected, "block {b}'s largest lands on 6");
        }
    }
    assert_eq!(
        row_scales,
        vec![4.0; rows],
        "each row under its largest block's 2^2"
    );
    assert_eq!(read_u32(&client, &quantized.scales.lossy_blocks), vec![1]);
}
