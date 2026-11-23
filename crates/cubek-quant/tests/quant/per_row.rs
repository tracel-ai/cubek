//! FP8-row, and block scales factored by row: that each launch writes the levels its scheme
//! names, and that the values they reconstruct are the input's, to the format's rounding, for
//! MXFP4, MXFP8 and wider blocks alike.

use cubecl::ir::FloatKind;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::{
    client::Client,
    ir::{ElemType, UIntKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_quant::per_row::{
    PerRowRefusal, factor_block_scales_by_row, quantize_fp8_per_row, quantize_row_factored,
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

    let quantized = quantize_fp8_per_row(
        &client,
        &input,
        f32::elem_type_native(),
        &QuantScheme::fp8_per_row(),
    )
    .expect("FP8-row");

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

/// `e4m3`, what the products these weights meet decode into.
const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);

/// The blocks of `scheme`'s level, `size` values each, under `ue8m0` scales.
fn blocks_of(value: QuantValue, size: u8) -> QuantScheme {
    QuantScheme::default()
        .with_value(value)
        .per_block([size], ScaleDtype::UE8M0)
        .with_store(QuantStore::PackedU32(0))
}

/// Each row's largest block scale becomes its `f32` scale and every block its offset below it;
/// the blocks more than eight powers of two below are counted, past which an `e2m1` value is no
/// `e4m3` one, and one past a `ue8m0`'s range flushes to the smallest offset.
#[test]
fn factoring_block_scales_by_row_stores_offsets_below_the_largest() {
    let client = cubecl::test_device().client();
    let rows = vec![
        vec![127, 125, 130, 128, 122, 123, 130, 130],
        vec![100, 100, 100, 100, 91, 92, 90, 100],
        vec![200, 2, 199, 200, 200, 200, 200, 200],
    ];
    let scales = scale_bytes(&client, &rows);

    let restated =
        factor_block_scales_by_row(&client, &blocks_of(QuantValue::E2M1, 32), &scales, E4M3)
            .expect("ue8m0 blocks");

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
            if below > 8 {
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

/// Block scales that are no powers of two are refused: factored by row, a block's would be its
/// quotient by its row's, which no `ue4m3` holds.
#[test]
fn factoring_refuses_block_scales_that_are_no_powers_of_two() {
    let client = cubecl::test_device().client();
    let scales = scale_bytes(&client, &[vec![56; 8]]);
    let nvfp4 = QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([16], ScaleDtype::UE4M3)
        .with_store(QuantStore::PackedU32(0));
    assert_eq!(
        factor_block_scales_by_row(&client, &nvfp4, &scales, E4M3).err(),
        Some(PerRowRefusal::BlockScalesNotPowersOfTwo)
    );
}

/// `rows × blocks` blocks of `block` values, each block's largest `largest_of(block)`, one block
/// of zeros at row 1, block 3.
fn blocked(
    rows: usize,
    blocks: usize,
    block: usize,
    largest_of: impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut data = Vec::new();
    for r in 0..rows {
        for b in 0..blocks {
            match (r, b) {
                (1, 3) => data.extend(vec![0.0; block]),
                _ => data.extend(row(block, largest_of(b), b + r)),
            }
        }
    }
    data
}

/// How a format's codes read, and its largest step: between its two largest values.
struct Format {
    decode: fn(u32) -> f32,
    largest_step: f32,
}

/// A float weight quantized under `scheme`, three rows, its block scales factored by row: each
/// value under the two levels its input's to within half the format's largest step under the
/// block's scale, and each block's largest landing on the format's. The rows' largest blocks are
/// under `2^2`, five powers of two above their least; `lossy` blocks are counted.
fn check_row_factored(scheme: QuantScheme, block: usize, k: usize, format: Format, lossy: u32) {
    let client = cubecl::test_device().client();
    let rows = 3;
    let blocks = k / block;
    let largest = scheme.value.range().1;
    let largest_of = |b: usize| largest * 2f32.powi((b % 6) as i32 - 3);
    let data = blocked(rows, blocks, block, largest_of);
    let input = f32_tensor(&client, &data, shape![rows, k]);

    let quantized = quantize_row_factored(&client, &input, f32::elem_type_native(), &scheme, E4M3)
        .expect("ue8m0 blocks");

    let codes = read_u32(&client, &quantized.codes);
    let offsets = read_bytes(&client, &quantized.scales.block_scales);
    let row_scales = read_f32(&client, &quantized.scales.row_scales);
    let (per_word, bits) = (scheme.num_quants(), scheme.value.size_bits());
    let mask = (1u32 << bits) - 1;
    for (at, &expected) in data.iter().enumerate() {
        let (r, b) = (at / k, at / block);
        let code = (codes[at / per_word] >> (bits * (at % per_word))) & mask;
        let scale = ue8m0(offsets[b]) * row_scales[r];
        let got = (format.decode)(code) * scale;
        assert!(
            (got - expected).abs() <= format.largest_step / 2.0 * scale,
            "value {at}: {got} for {expected}"
        );
        if expected == largest_of(b % blocks) {
            assert_eq!(got, expected, "block {b}'s largest lands on the format's");
        }
    }
    assert_eq!(
        row_scales,
        vec![4.0; rows],
        "each row under its largest block's 2^2"
    );
    assert_eq!(
        read_u32(&client, &quantized.scales.lossy_blocks),
        vec![lossy]
    );
}

const E2M1_FORMAT: Format = Format {
    decode: e2m1,
    largest_step: 2.0,
};

/// MXFP4, three rows of twelve blocks of 32: only the block of zeros is lossy, every other within
/// the eight powers of two an `e2m1` value keeps in `e4m3`.
#[test]
fn an_mxfp4_weight_quantizes_row_factored() {
    check_row_factored(blocks_of(QuantValue::E2M1, 32), 32, 384, E2M1_FORMAT, 1);
}

/// `e2m1` codes under blocks of 64, three rows of twelve: only the block of zeros is lossy.
#[test]
fn an_e2m1_weight_of_64_wide_blocks_quantizes_row_factored() {
    check_row_factored(blocks_of(QuantValue::E2M1, 64), 64, 768, E2M1_FORMAT, 1);
}

/// MXFP8, `e4m3` codes under blocks of 32, three rows of twelve: every block below its row's
/// largest is lossy, its smallest subnormals past `e4m3`'s, ten a row.
#[test]
fn an_mxfp8_weight_quantizes_row_factored() {
    let format = Format {
        decode: |code| e4m3(code as u8),
        largest_step: 32.0,
    };
    check_row_factored(blocks_of(QuantValue::E4M3, 32), 32, 384, format, 30);
}
