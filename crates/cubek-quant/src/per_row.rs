//! Per-row scales for an 8-bit float product: FP8-row activations
//! ([`QuantScheme::fp8_per_row`](crate::scheme::QuantScheme::fp8_per_row)) against MXFP4
//! weights with row-factored scales
//! ([`QuantScheme::mxfp4_row_factored`](crate::scheme::QuantScheme::mxfp4_row_factored)).
//!
//! A product of the two contracts `e4m3` values on the device's plain `e4m3` instruction: each
//! row's scale comes out of the sum, applied once to its output, and the weight's codes decode
//! into `e4m3` under their block's offset as they load. Each step here is one launch the caller
//! runs, on a matrix of rows with `k` innermost, and none reads anything back: an MXFP4 weight's
//! scales factored by row once as it loads ([`factor_mxfp4_scales_by_row`]), a float weight
//! quantized straight to that ([`quantize_mxfp4_row_factored`]), an activation quantized a row at
//! a time before each product ([`quantize_fp8_per_row`]).

use cubecl::{
    calculate_cube_count_elemwise,
    ir::{ElemType, FloatKind, UIntKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};

use crate::quantize::pack_q;
use crate::scheme::QuantValue;

/// Values a block of an MXFP4 weight holds under one scale.
pub const MXFP4_BLOCK: usize = 32;

/// The lowest offset of a block's scale below its row's, as a power of two, at which every
/// `e2m1` value of the block stays an `e4m3` one: `0.5 × 2^-8` is `e4m3`'s smallest subnormal.
/// The decode into `e4m3` rounds a block further below, which the values themselves do not, and
/// [`RowFactoredScales::lossy_blocks`] counts it.
pub const EXACT_LOWEST_OFFSET: i32 = -8;

/// Units of a cube, a row's: each reads its strided share of the row.
const ROW_UNITS: usize = 64;
/// `e4m3` values, a byte each, a `u32` word holds.
const E4M3_PER_WORD: usize = 4;
/// `e2m1` codes, four bits each, a `u32` word holds.
const E2M1_PER_WORD: usize = 8;
/// `ue8m0` block scales, a byte each, a `u32` word holds.
const SCALES_PER_WORD: usize = 4;
/// The bits of one byte of a word, and those bits set.
const BYTE_BITS: u32 = u8::BITS;
const BYTE_MASK: u32 = u8::MAX as u32;
/// `e4m3`'s largest value, what a row's largest magnitude quantizes to.
const E4M3_LARGEST: f32 = 448.0;
/// `e2m1`'s largest value, what a block's largest magnitude quantizes under.
const E2M1_LARGEST: f32 = 6.0;
/// The bias of an `f32` exponent field, which a `ue8m0` byte shares.
const EXPONENT_BIAS: u32 = 127;
/// The mantissa bits an `f32` stores, below its exponent field.
const F32_STORED_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
/// The largest finite `ue8m0` byte, `2^127`: `0xFF` is its NaN.
const UE8M0_LARGEST: u32 = 254;

/// An activation quantized to FP8-row.
pub struct Fp8RowActivation {
    /// Its `e4m3` values, `[rows, k]`, each its value over its row's scale.
    pub values: TensorHandle,
    /// Its scales, `[rows]`, one `f32` a row.
    pub scales: TensorHandle,
}

/// `input`, a contiguous float `[rows, k]` of `input_elem`, quantized to FP8-row: each row's
/// scale its largest magnitude over `e4m3`'s largest, one where the row is zero, and each value
/// over its row's scale rounded to the nearest `e4m3`.
///
/// # Panics
///
/// Where `k` is not a multiple of four, the values a word of the output holds.
pub fn quantize_fp8_per_row(
    client: &Client,
    input: &TensorHandle,
    input_elem: ElemType,
) -> Fp8RowActivation {
    let (rows, k) = rows_of(input.shape());
    assert!(
        k.is_multiple_of(E4M3_PER_WORD),
        "quantize_fp8_per_row: k ({k}) should hold whole words of four e4m3 values"
    );
    let words = TensorHandle::empty(
        client,
        [rows, k / E4M3_PER_WORD],
        ElemType::UInt(UIntKind::U32),
    );
    let scales = TensorHandle::empty(client, [rows], ElemType::Float(FloatKind::F32));
    let address_type = input.required_address_type();
    // SAFETY: one cube a row, which reads and writes that row alone; a cube past the rows leaves.
    unsafe {
        cube_fp8_per_row_quantize::launch_unchecked(
            client,
            calculate_cube_count_elemwise(client, rows, CubeDim::new_single()),
            CubeDim::new_1d(ROW_UNITS as u32),
            address_type,
            input.clone().into_arg(),
            words.clone().into_arg(),
            scales.clone().into_arg(),
            input_elem,
        )
    };
    let values =
        TensorHandle::new_contiguous([rows, k], words.handle, ElemType::Float(FloatKind::E4M3));
    Fp8RowActivation { values, scales }
}

/// An MXFP4 weight's scales factored by row: the same values as MXFP4. Each row's scale is its
/// largest block scale, an `f32`, and each block scale is stored as its offset below it, the same
/// byte a block. It adds no precision; it lets a decode into `e4m3` fold the block scales in
/// exactly. The weight's codes are unchanged.
pub struct RowFactoredScales {
    /// `ue8m0` offsets, `[rows, k / 32]`, each `2^(s - largest)`: no longer the checkpoint's bytes,
    /// though under the row's scale they are its scales.
    pub block_scales: TensorHandle,
    /// One `f32` a row, its largest block scale.
    pub row_scales: TensorHandle,
    /// One `u32`: the blocks whose offset is below [`EXACT_LOWEST_OFFSET`], which may round as a
    /// product decodes them into `e4m3`, the decode's limit and not the format's. Zero for a weight
    /// the decode takes exactly; a block of zeros, under the smallest scale, counts though it
    /// decodes exactly.
    pub lossy_blocks: TensorHandle,
}

/// The `ue8m0` block scales of an MXFP4 weight, `[rows, k / 32]` bytes, factored by row
/// ([`RowFactoredScales`]): the same values, each row's largest block scale beside each block's
/// offset below it.
///
/// # Panics
///
/// Where a row's scales are not whole words of four, `k` a multiple of 128.
pub fn factor_mxfp4_scales_by_row(
    client: &Client,
    block_scales: &TensorHandle,
) -> RowFactoredScales {
    let (rows, blocks) = rows_of(block_scales.shape());
    assert!(
        blocks.is_multiple_of(SCALES_PER_WORD),
        "factor_mxfp4_scales_by_row: a row's {blocks} block scales should be whole words of four"
    );
    let word_shape = [rows, blocks / SCALES_PER_WORD];
    let words = TensorHandle::new_contiguous(
        word_shape,
        block_scales.handle.clone(),
        ElemType::UInt(UIntKind::U32),
    );
    let offsets = TensorHandle::empty(client, word_shape, ElemType::UInt(UIntKind::U32));
    let row_scales = TensorHandle::empty(client, [rows], ElemType::Float(FloatKind::F32));
    let lossy_blocks = TensorHandle::zeros(client, [1], ElemType::UInt(UIntKind::U32));
    let address_type = words.required_address_type();
    // SAFETY: one cube a row of scales, which reads and writes that row alone and adds into the
    // one counter atomically; a cube past the rows leaves.
    unsafe {
        cube_mxfp4_scales_factor_by_row::launch_unchecked(
            client,
            calculate_cube_count_elemwise(client, rows, CubeDim::new_single()),
            CubeDim::new_1d(ROW_UNITS as u32),
            address_type,
            words.into_arg(),
            offsets.clone().into_arg(),
            row_scales.clone().into_arg(),
            lossy_blocks.clone().into_arg().into_buffer_arg(),
        )
    };
    RowFactoredScales {
        block_scales: TensorHandle::new_contiguous(
            [rows, blocks],
            offsets.handle,
            ElemType::UInt(UIntKind::U8),
        ),
        row_scales,
        lossy_blocks,
    }
}

/// A float weight quantized to MXFP4 with its scales factored by row.
pub struct Mxfp4RowFactoredWeight {
    /// Its `e2m1` codes, `[rows, k / 8]` words, eight a word, the lowest first.
    pub codes: TensorHandle,
    /// Its scales.
    pub scales: RowFactoredScales,
}

/// `input`, a contiguous float `[rows, k]` of `input_elem`, quantized to MXFP4, then its scales
/// factored by row ([`factor_mxfp4_scales_by_row`]). Each block's `ue8m0` scale is its largest
/// magnitude over `e2m1`'s largest, rounded up to a power of two, so no value saturates where the
/// OCP MX rule rounds down and saturates the largest; each value is its nearest code under it.
///
/// # Panics
///
/// Where `k` is not a multiple of 128, whole words of four block scales.
pub fn quantize_mxfp4_row_factored(
    client: &Client,
    input: &TensorHandle,
    input_elem: ElemType,
) -> Mxfp4RowFactoredWeight {
    let (rows, k) = rows_of(input.shape());
    let per_word = SCALES_PER_WORD * MXFP4_BLOCK;
    assert!(
        k.is_multiple_of(per_word),
        "quantize_mxfp4_row_factored: k ({k}) should hold whole words of four block scales"
    );
    let codes = TensorHandle::empty(
        client,
        [rows, k / E2M1_PER_WORD],
        ElemType::UInt(UIntKind::U32),
    );
    let scales = TensorHandle::empty(client, [rows, k / per_word], ElemType::UInt(UIntKind::U32));
    let units = rows * k / per_word;
    let cube_dim = CubeDim::new(client, units);
    let address_type = input.required_address_type();
    // SAFETY: one unit a word of block scales, which reads its four blocks and writes their codes
    // and scales alone; a unit past the words leaves.
    unsafe {
        unit_mxfp4_quantize::launch_unchecked(
            client,
            calculate_cube_count_elemwise(client, units, cube_dim),
            cube_dim,
            address_type,
            input.clone().into_arg(),
            codes.clone().into_arg(),
            scales.clone().into_arg(),
            input_elem,
        )
    };
    let block_scales = TensorHandle::new_contiguous(
        [rows, k / MXFP4_BLOCK],
        scales.handle,
        ElemType::UInt(UIntKind::U8),
    );
    Mxfp4RowFactoredWeight {
        codes,
        scales: factor_mxfp4_scales_by_row(client, &block_scales),
    }
}

/// A matrix's rows and the extent of each, its innermost dimension.
fn rows_of(shape: &Shape) -> (usize, usize) {
    let inner = shape[shape.rank() - 1];
    (shape.num_elements() / inner.max(1), inner)
}

/// One cube a row: the row's largest magnitude, reduced across the cube, then each unit's words
/// of four values written under it.
///
/// - `input`: `[rows, k]` floats of `F`, contiguous.
/// - `words`: `[rows, k / 4]`, the `e4m3` values four a word, the lowest first.
/// - `scales`: `[rows]`, each row's `f32` scale.
#[cube(launch_unchecked, address_type = "dynamic")]
fn cube_fp8_per_row_quantize<F: Float>(
    input: &Tensor<F>,
    words: &mut Tensor<u32>,
    scales: &mut Tensor<f32>,
    #[define(F)] _float: ElemType,
) {
    let row = CUBE_POS;
    if row >= scales.len() {
        terminate!();
    }
    let word_count = words.shape(1);
    let start = row * word_count * E4M3_PER_WORD;
    let mut largest = 0f32;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        #[unroll]
        for j in 0..E4M3_PER_WORD {
            let value = f32::cast_from(input[start + w * E4M3_PER_WORD + j]);
            largest = f32::max(largest, f32::abs(value));
        }
    }
    // A magnitude's bits order as the magnitude does.
    let largest = f32::reinterpret(cube_max(u32::reinterpret(largest)));
    let scale = select(largest > 0f32, largest / E4M3_LARGEST, 1f32);
    if UNIT_POS == 0 {
        scales[row] = scale;
    }
    let size!(N) = E4M3_PER_WORD;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        let mut quad = Vector::<f32, N>::empty();
        #[unroll]
        for j in 0..E4M3_PER_WORD {
            quad.insert(
                j,
                f32::cast_from(input[start + w * E4M3_PER_WORD + j]) / scale,
            );
        }
        words[row * word_count + w] = pack_q::<f32, N, u32>(quad, QuantValue::E4M3);
    }
}

/// One cube a row of block scales: the row's largest, reduced across the cube, then each word of
/// four offsets below it. Unit zero writes the row's scale and adds the row's lossy blocks into
/// the counter.
///
/// - `scales`: `[rows, k / 128]`, the `ue8m0` block scales four a word.
/// - `offsets`: `[rows, k / 128]`, each block's offset below its row's largest, as a `ue8m0`.
/// - `row_scales`: `[rows]`, each row's largest block scale as an `f32`.
/// - `lossy`: one counter, the blocks below [`EXACT_LOWEST_OFFSET`].
#[cube(launch_unchecked, address_type = "dynamic")]
fn cube_mxfp4_scales_factor_by_row(
    scales: &Tensor<u32>,
    offsets: &mut Tensor<u32>,
    row_scales: &mut Tensor<f32>,
    lossy: &[Atomic<u32>],
) {
    let row = CUBE_POS;
    if row >= row_scales.len() {
        terminate!();
    }
    let word_count = scales.shape(1);
    let start = row * word_count;
    let mut largest = 0u32;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        let word = scales[start + w];
        #[unroll]
        for j in 0..SCALES_PER_WORD {
            largest = u32::max(largest, byte_of(word, j));
        }
    }
    let largest = cube_max(largest);
    let mut lossy_here = 0u32;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        let word = scales[start + w];
        let mut factored = 0u32;
        #[unroll]
        for j in 0..SCALES_PER_WORD {
            let below = largest - byte_of(word, j);
            // An offset past a `ue8m0`'s range flushes to its smallest, and is lossy.
            let offset = select(below > EXPONENT_BIAS, 0u32, EXPONENT_BIAS - below);
            lossy_here += u32::cast_from(below > (-EXACT_LOWEST_OFFSET) as u32);
            factored |= offset << comptime!(BYTE_BITS * j as u32);
        }
        offsets[start + w] = factored;
    }
    let lossy_in_row = cube_sum(lossy_here);
    if UNIT_POS == 0 {
        row_scales[row] = power_of_two(largest);
        lossy[0].fetch_add(lossy_in_row);
    }
}

/// A unit a word of four block scales, 128 values: each block's scale, then its 32 codes, four
/// words of eight.
///
/// - `input`: `[rows, k]` floats of `F`, contiguous.
/// - `codes`: `[rows, k / 8]`, the `e2m1` codes eight a word, the lowest first.
/// - `scales`: `[rows, k / 128]`, the `ue8m0` block scales four a word.
#[cube(launch_unchecked, address_type = "dynamic")]
fn unit_mxfp4_quantize<F: Float>(
    input: &Tensor<F>,
    codes: &mut Tensor<u32>,
    scales: &mut Tensor<u32>,
    #[define(F)] _float: ElemType,
) {
    if ABSOLUTE_POS >= scales.len() {
        terminate!();
    }
    let start = ABSOLUTE_POS * SCALES_PER_WORD * MXFP4_BLOCK;
    let size!(N) = E2M1_PER_WORD;
    let mut scale_word = 0u32;
    #[unroll]
    for b in 0..SCALES_PER_WORD {
        let block = start + b * MXFP4_BLOCK;
        let mut largest = 0f32;
        #[unroll]
        for j in 0..MXFP4_BLOCK {
            largest = f32::max(largest, f32::abs(f32::cast_from(input[block + j])));
        }
        let byte = ue8m0_at_least(largest / E2M1_LARGEST);
        scale_word |= byte << comptime!(BYTE_BITS * b as u32);
        // A zero block's codes are zero whatever its scale.
        let inverse = select(byte == 0, 0f32, power_of_two(2 * EXPONENT_BIAS - byte));
        #[unroll]
        for w in 0..comptime!(MXFP4_BLOCK / E2M1_PER_WORD) {
            let mut eight = Vector::<f32, N>::empty();
            #[unroll]
            for j in 0..E2M1_PER_WORD {
                let value = f32::cast_from(input[block + w * E2M1_PER_WORD + j]);
                eight.insert(j, value * inverse);
            }
            codes[block / E2M1_PER_WORD + w] = pack_q::<f32, N, u32>(eight, QuantValue::E2M1);
        }
    }
    scales[ABSOLUTE_POS] = scale_word;
}

/// The `j`-th byte of `word`, the lowest first.
#[cube]
fn byte_of(word: u32, #[comptime] j: usize) -> u32 {
    (word >> comptime!(BYTE_BITS * j as u32)) & BYTE_MASK
}

/// The `ue8m0` byte of the smallest power of two at least `value`, a positive `f32`: its exponent
/// field, one more where a mantissa bit is set, at most the largest finite one; zero for zero.
#[cube]
fn ue8m0_at_least(value: f32) -> u32 {
    let bits = u32::reinterpret(value);
    let exponent = bits >> F32_STORED_MANTISSA_BITS;
    let mantissa = bits & ((1u32 << F32_STORED_MANTISSA_BITS) - 1);
    u32::min(exponent + u32::cast_from(mantissa != 0), UE8M0_LARGEST)
}

/// `2^(byte - 127)` as an `f32`, `byte` a `ue8m0` one: its exponent field, or, for a byte of zero,
/// the subnormal `2^-127`.
#[cube]
fn power_of_two(byte: u32) -> f32 {
    let bits = select(
        byte > 0,
        byte << F32_STORED_MANTISSA_BITS,
        1u32 << (F32_STORED_MANTISSA_BITS - 1),
    );
    f32::reinterpret(bits)
}

/// The largest of `value` across the cube's units, every unit given it.
#[cube]
fn cube_max(value: u32) -> u32 {
    cube_fold(value, Fold::Max)
}

/// The sum of `value` across the cube's units, every unit given it.
#[cube]
fn cube_sum(value: u32) -> u32 {
    cube_fold(value, Fold::Sum)
}

/// How [`cube_fold`] combines two units' values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Fold {
    Max,
    Sum,
}

/// `value` folded across the [`ROW_UNITS`] units of the cube by `fold`, in halving strides
/// through shared memory, every unit given the result.
#[cube]
fn cube_fold(value: u32, #[comptime] fold: Fold) -> u32 {
    let mut shared = Shared::<[u32]>::new_slice(ROW_UNITS);
    let unit = UNIT_POS as usize;
    shared[unit] = value;
    sync_cube();
    #[unroll]
    for level in 0..comptime!(ROW_UNITS.trailing_zeros()) {
        let stride = comptime!(ROW_UNITS >> (level + 1));
        if unit < stride {
            shared[unit] = match fold {
                Fold::Max => u32::max(shared[unit], shared[unit + stride]),
                Fold::Sum => shared[unit] + shared[unit + stride],
            };
        }
        sync_cube();
    }
    let folded = shared[0];
    sync_cube();
    folded
}
