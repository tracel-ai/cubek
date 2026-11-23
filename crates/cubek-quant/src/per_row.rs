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
    ir::{ElemType, FloatKind, UIntKind, types::Fp8Format},
    post_processing::{fp4::float_to_e2m1_bits, minifloat::f32_to_fp8_bits},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};

/// Values a block of an MXFP4 weight holds under one scale.
pub const MXFP4_BLOCK: usize = 32;

/// The lowest offset of a block's scale below its row's, as a power of two, at which every
/// `e2m1` value of the block stays an `e4m3` one: `0.5 × 2^-8` is `e4m3`'s smallest subnormal.
/// The decode into `e4m3` rounds a block further below, which the values themselves do not, and
/// [`RowFactoredScales::lossy_blocks`] counts it.
pub const EXACT_LOWEST_OFFSET: i32 = -8;

/// Units of a cube, a row's: each reads its strided share of the row.
const ROW_UNITS: u32 = 64;
/// `e4m3`'s largest value, what a row's largest magnitude quantizes to.
const E4M3_LARGEST: f32 = 448.0;
/// `e2m1`'s largest value, what a block's largest magnitude quantizes under.
const E2M1_LARGEST: f32 = 6.0;
/// The bias of an `f32` exponent, and of a `ue8m0` scale, which is one.
const EXPONENT_BIAS: u32 = 127;
const F32_MANTISSA_BITS: u32 = 23;

/// An activation quantized to FP8-row: `e4m3` values, `[rows, k]`, and one `f32` scale a row.
pub struct Fp8PerRow {
    pub values: TensorHandle,
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
) -> Fp8PerRow {
    let (rows, k) = rows_of(input.shape());
    assert!(
        k.is_multiple_of(4),
        "quantize_fp8_per_row: k ({k}) holds whole words of four e4m3 values"
    );
    let words = TensorHandle::empty(client, [rows, k / 4], ElemType::UInt(UIntKind::U32));
    let scales = TensorHandle::empty(client, [rows], ElemType::Float(FloatKind::F32));
    let cube_count = calculate_cube_count_elemwise(client, rows, CubeDim::new_single());
    fp8_per_row_kernel::launch(
        client,
        cube_count,
        CubeDim::new_1d(ROW_UNITS),
        input.clone().into_arg(),
        words.clone().into_arg(),
        scales.clone().into_arg(),
        input_elem,
    );
    let values =
        TensorHandle::new_contiguous([rows, k], words.handle, ElemType::Float(FloatKind::E4M3));
    Fp8PerRow { values, scales }
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
        blocks.is_multiple_of(4),
        "factor_mxfp4_scales_by_row: a row's {blocks} block scales are whole words of four"
    );
    let words = |handle: &TensorHandle| {
        TensorHandle::new_contiguous(
            [rows, blocks / 4],
            handle.handle.clone(),
            ElemType::UInt(UIntKind::U32),
        )
    };
    let offsets = TensorHandle::empty(client, [rows, blocks / 4], ElemType::UInt(UIntKind::U32));
    let row_scales = TensorHandle::empty(client, [rows], ElemType::Float(FloatKind::F32));
    let lossy_blocks = TensorHandle::zeros(client, [1], ElemType::UInt(UIntKind::U32));
    let cube_count = calculate_cube_count_elemwise(client, rows, CubeDim::new_single());
    factor_scales_by_row_kernel::launch(
        client,
        cube_count,
        CubeDim::new_1d(ROW_UNITS),
        words(block_scales).into_arg(),
        offsets.clone().into_arg(),
        row_scales.clone().into_arg(),
        lossy_blocks.clone().into_arg().into_buffer_arg(),
        ROW_UNITS as usize,
    );
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

/// A float weight quantized to MXFP4 with its scales factored by row: its `e2m1` codes,
/// `[rows, k / 8]` words, eight a word the lowest first, and its scales ([`RowFactoredScales`]).
pub struct QuantizedMxfp4RowFactored {
    pub codes: TensorHandle,
    pub scales: RowFactoredScales,
}

/// `input`, a contiguous float `[rows, k]` of `input_elem`, quantized to MXFP4 as burn's
/// `quantize_dynamic` does, each block's `ue8m0` scale its largest magnitude over `e2m1`'s
/// largest rounded up to a power of two, then its scales factored by row
/// ([`factor_mxfp4_scales_by_row`]).
///
/// # Panics
///
/// Where `k` is not a multiple of 128, whole words of four block scales.
pub fn quantize_mxfp4_row_factored(
    client: &Client,
    input: &TensorHandle,
    input_elem: ElemType,
) -> QuantizedMxfp4RowFactored {
    let (rows, k) = rows_of(input.shape());
    assert!(
        k.is_multiple_of(4 * MXFP4_BLOCK),
        "quantize_mxfp4_row_factored: k ({k}) holds whole words of four block scales"
    );
    let codes = TensorHandle::empty(client, [rows, k / 8], ElemType::UInt(UIntKind::U32));
    let scales = TensorHandle::empty(
        client,
        [rows, k / (4 * MXFP4_BLOCK)],
        ElemType::UInt(UIntKind::U32),
    );
    // A unit a word of scales: four blocks.
    let units = rows * k / (4 * MXFP4_BLOCK);
    let cube_dim = CubeDim::new(client, units);
    let cube_count = calculate_cube_count_elemwise(client, units, cube_dim);
    mxfp4_kernel::launch(
        client,
        cube_count,
        cube_dim,
        input.clone().into_arg(),
        codes.clone().into_arg(),
        scales.clone().into_arg(),
        input_elem,
    );
    let block_scales = TensorHandle::new_contiguous(
        [rows, k / MXFP4_BLOCK],
        scales.handle,
        ElemType::UInt(UIntKind::U8),
    );
    QuantizedMxfp4RowFactored {
        codes,
        scales: factor_mxfp4_scales_by_row(client, &block_scales),
    }
}

/// A matrix's rows and the extent of each, its innermost dimension.
fn rows_of(shape: &Shape) -> (usize, usize) {
    let inner = shape[shape.rank() - 1];
    (shape.num_elements() / inner.max(1), inner)
}

/// One cube a row: its largest magnitude reduced across the cube, then each unit's words of four
/// values written under it.
#[cube(launch)]
fn fp8_per_row_kernel<F: Float>(
    input: &Tensor<F>,
    words: &mut Tensor<u32>,
    scales: &mut Tensor<f32>,
    #[define(F)] _input: ElemType,
) {
    let row = CUBE_POS;
    if row >= scales.len() {
        terminate!();
    }
    let word_count = words.shape(1);
    let start = row * word_count * 4;
    let mut largest = 0f32;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        #[unroll]
        for j in 0..4usize {
            largest = f32::max(largest, f32::abs(f32::cast_from(input[start + w * 4 + j])));
        }
    }
    // A magnitude's bits order as the magnitude does.
    let largest = f32::reinterpret(cube_max(u32::reinterpret(largest), ROW_UNITS as usize));
    let scale = select(largest > 0f32, largest / E4M3_LARGEST, 1f32);
    if UNIT_POS == 0 {
        scales[row] = scale;
    }
    let size!(N4) = 4usize;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        let mut quads = Vector::<f32, N4>::empty();
        #[unroll]
        for j in 0..4usize {
            quads.insert(j, f32::cast_from(input[start + w * 4 + j]) / scale);
        }
        let bytes = f32_to_fp8_bits::<N4>(quads, Fp8Format::E4M3);
        let mut word = 0u32;
        #[unroll]
        for j in 0..4usize {
            word |= (bytes.extract(j) & 0xFFu32) << comptime!(8 * j as u32);
        }
        words[row * word_count + w] = word;
    }
}

/// One cube a row of block scales: its largest, then each word of four offsets below it, the
/// blocks past [`EXACT_LOWEST_OFFSET`] counted once a unit.
#[cube(launch)]
fn factor_scales_by_row_kernel(
    scales: &Tensor<u32>,
    offsets: &mut Tensor<u32>,
    row_scales: &mut Tensor<f32>,
    lossy: &[Atomic<u32>],
    #[comptime] units: usize,
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
        for j in 0..4u32 {
            largest = u32::max(largest, (word >> (8 * j)) & 0xFFu32);
        }
    }
    let largest = cube_max(largest, units);
    if UNIT_POS == 0 {
        // `2^(largest - 127)`: its `f32` exponent field, or the subnormal of `2^-127`.
        let bits = select(
            largest > 0,
            largest << F32_MANTISSA_BITS,
            1u32 << (F32_MANTISSA_BITS - 1),
        );
        row_scales[row] = f32::reinterpret(bits);
    }
    let mut lossy_here = 0u32;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        let word = scales[start + w];
        let mut restated = 0u32;
        #[unroll]
        for j in 0..4u32 {
            let below = largest - ((word >> (8 * j)) & 0xFFu32);
            // An offset past a `ue8m0`'s range flushes to its smallest, and is lossy.
            let offset = select(below > EXPONENT_BIAS, 0u32, EXPONENT_BIAS - below);
            lossy_here += u32::cast_from(below > (-EXACT_LOWEST_OFFSET) as u32);
            restated |= offset << (8 * j);
        }
        offsets[start + w] = restated;
    }
    if lossy_here > 0 {
        lossy[0].fetch_add(lossy_here);
    }
}

/// A unit a word of four block scales, 128 values: each block's scale, then its 32 codes, four
/// words of eight.
#[cube(launch)]
fn mxfp4_kernel<F: Float>(
    input: &Tensor<F>,
    codes: &mut Tensor<u32>,
    scales: &mut Tensor<u32>,
    #[define(F)] _input: ElemType,
) {
    if ABSOLUTE_POS >= scales.len() {
        terminate!();
    }
    let start = ABSOLUTE_POS * 4 * MXFP4_BLOCK;
    let size!(N8) = 8usize;
    let mut scale_word = 0u32;
    #[unroll]
    for b in 0..4usize {
        let block = start + b * MXFP4_BLOCK;
        let mut largest = 0f32;
        #[unroll]
        for j in 0..MXFP4_BLOCK {
            largest = f32::max(largest, f32::abs(f32::cast_from(input[block + j])));
        }
        // Rounded up to a power of two: its exponent, one more where a mantissa bit is set, which
        // is the `ue8m0` byte of the same power. A zero block takes the smallest scale.
        let bits = u32::reinterpret(largest / E2M1_LARGEST);
        let exponent = (bits >> F32_MANTISSA_BITS) & 0xFFu32;
        let mantissa = bits & ((1u32 << F32_MANTISSA_BITS) - 1);
        let byte = u32::min(exponent + u32::cast_from(mantissa != 0), 254u32);
        scale_word |= byte << comptime!(8 * b as u32);
        let inverse = f32::reinterpret((2 * EXPONENT_BIAS - byte) << F32_MANTISSA_BITS);
        let inverse = select(byte == 0, 0f32, inverse);
        #[unroll]
        for w in 0..comptime!(MXFP4_BLOCK / 8) {
            let mut eight = Vector::<f32, N8>::empty();
            #[unroll]
            for j in 0..8usize {
                eight.insert(j, f32::cast_from(input[block + w * 8 + j]) * inverse);
            }
            let fields = float_to_e2m1_bits::<f32, N8>(eight);
            let mut word = 0u32;
            #[unroll]
            for j in 0..8usize {
                word |= (fields.extract(j) & 0xFu32) << comptime!(4 * j as u32);
            }
            codes[(block + w * 8) / 8] = word;
        }
    }
    scales[ABSOLUTE_POS] = scale_word;
}

/// The largest of `value` across the cube's `units`, every unit given it, through shared memory.
#[cube]
fn cube_max(value: u32, #[comptime] units: usize) -> u32 {
    let mut shared = Shared::<[u32]>::new_slice(units);
    let unit = UNIT_POS as usize;
    shared[unit] = value;
    sync_cube();
    #[unroll]
    for level in 0..comptime!(units.trailing_zeros()) {
        let stride = comptime!(units >> (level + 1));
        if unit < stride {
            shared[unit] = u32::max(shared[unit], shared[unit + stride]);
        }
        sync_cube();
    }
    let largest = shared[0];
    sync_cube();
    largest
}
