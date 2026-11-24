//! Per-row scales for an 8-bit float product: activations under one scale a row (FP8-row,
//! [`QuantScheme::fp8_per_row`]) against block-scaled weights whose power-of-two block scales are
//! factored by row ([`QuantScheme::mxfp4_row_factored`] among them).
//!
//! A product of the two contracts the activation's 8-bit values on the device's plain
//! instruction for them: each row's scale comes out of the sum, applied once to its output, and
//! the weight's codes decode into the activation's element under their block's offset as they
//! load. Each step here is one launch the caller runs, on a matrix of rows with `k` innermost,
//! and none reads anything back: a weight's block scales factored by row once as it loads
//! ([`factor_block_scales_by_row`]), a float weight quantized straight to that
//! ([`quantize_row_factored`]), an activation quantized a row at a time before each product
//! ([`quantize_fp8_per_row`]). Every count, value and bound is the scheme's.

use cubecl::{
    calculate_cube_count_elemwise,
    ir::{ElemType, FloatKind, UIntKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};

use crate::quantize::pack_q;
use crate::scheme::{QuantScheme, QuantValue, ScaleDtype};

/// Why a launch here refuses a scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerRowRefusal {
    /// The scheme's values are not 8-bit floats under one scale a row, its only level.
    NotRowScaled,
    /// The scheme's block scales are not powers of two (`ue8m0`): factored by row, each block's
    /// would be its quotient by the row's, which its dtype does not hold.
    BlockScalesNotPowersOfTwo,
    /// The scheme's blocks hold no whole words of its codes, or a row no whole word of its
    /// block scales.
    BlocksNotWholeWords,
}

/// Units of a cube, a row's: each reads its strided share of the row.
const ROW_UNITS: usize = 64;
/// `ue8m0` block scales, a byte each, a `u32` word holds.
const SCALES_PER_WORD: usize = 4;
/// The bits of one byte of a word, and those bits set.
const BYTE_BITS: u32 = u8::BITS;
const BYTE_MASK: u32 = u8::MAX as u32;
/// The bias of an `f32` exponent field, which a `ue8m0` byte shares.
const EXPONENT_BIAS: u32 = 127;
/// The mantissa bits an `f32` stores, below its exponent field.
const F32_STORED_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
/// The largest finite `ue8m0` byte, `2^127`: `0xFF` is its NaN.
const UE8M0_LARGEST: u32 = 254;

/// An activation quantized to FP8-row.
pub struct RowScaledActivation {
    /// Its 8-bit float values, `[rows, k]`, each its value over its row's scale.
    pub values: TensorHandle,
    /// Its scales, `[rows]`, one `f32` a row.
    pub scales: TensorHandle,
}

/// `input`, a float `[rows, k]` of `input_elem`, quantized under `scheme`, 8-bit float values
/// under one scale a row ([`QuantScheme::fp8_per_row`]): each row's scale its largest magnitude
/// over the format's largest, one where the row is zero, and each value over its row's scale
/// rounded to the format's nearest.
///
/// # Errors
///
/// [`PerRowRefusal::NotRowScaled`] for a scheme of other values or levels, and
/// [`PerRowRefusal::BlocksNotWholeWords`] where `k` holds no whole words of its values.
pub fn quantize_fp8_per_row(
    client: &Client,
    input: &TensorHandle,
    input_elem: ElemType,
    scheme: &QuantScheme,
) -> Result<RowScaledActivation, PerRowRefusal> {
    let element = match scheme.value {
        QuantValue::E4M3 => ElemType::Float(FloatKind::E4M3),
        QuantValue::E5M2 => ElemType::Float(FloatKind::E5M2),
        _ => return Err(PerRowRefusal::NotRowScaled),
    };
    if scheme.num_levels() != 1 || scheme.row_scale().is_none() {
        return Err(PerRowRefusal::NotRowScaled);
    }
    let (rows, k) = rows_of(input.shape());
    let per_word = scheme.num_quants();
    if !k.is_multiple_of(per_word) {
        return Err(PerRowRefusal::BlocksNotWholeWords);
    }
    // Flat, so no runtime pads its rows.
    let words = TensorHandle::empty(client, [rows * k / per_word], ElemType::UInt(UIntKind::U32));
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
            *scheme,
            input_elem,
        )
    };
    let values = TensorHandle::new_contiguous([rows, k], words.handle, element);
    Ok(RowScaledActivation { values, scales })
}

/// A block-scaled weight's scales factored by row: the same values as the scheme it was factored
/// from. Each row's scale is its largest block scale, an `f32`, and each block scale is stored
/// as its offset below it, the same byte a block. It adds no precision; it lets a decode into an
/// 8-bit float fold the block scales in exactly.
pub struct RowFactoredScales {
    /// `ue8m0` offsets, `[rows, k / block]`, each `2^(s - largest)`: no longer the checkpoint's
    /// bytes, though under the row's scale they are its scales.
    pub block_scales: TensorHandle,
    /// One `f32` a row, its largest block scale.
    pub row_scales: TensorHandle,
    /// One `u32`: the blocks whose offset is below the lowest at which every one of their values
    /// stays one of the element they decode into, which may round as a product decodes them, the
    /// decode's limit and not the format's. Zero for a weight the decode takes exactly; a block of
    /// zeros, under the smallest scale, counts though it decodes exactly.
    pub lossy_blocks: TensorHandle,
}

/// The `ue8m0` block scales of a weight quantized under `scheme`, `[rows, k / block]` bytes in
/// dense rows, factored by row ([`RowFactoredScales`]): the same values, each row's largest block
/// scale beside each block's offset below it. A block is counted lossy past the offset under
/// which its smallest nonzero value is no `decoded_into` value.
///
/// # Errors
///
/// [`PerRowRefusal::BlockScalesNotPowersOfTwo`] for a scheme whose block scales are no `ue8m0`,
/// and [`PerRowRefusal::BlocksNotWholeWords`] where a row holds no whole words of four.
///
/// # Panics
///
/// Where the block scales do not lie in dense rows, which the launch reads as words.
pub fn factor_block_scales_by_row(
    client: &Client,
    scheme: &QuantScheme,
    block_scales: &TensorHandle,
    decoded_into: ElemType,
) -> Result<RowFactoredScales, PerRowRefusal> {
    if scheme.block_scale().map(|block| block.dtype) != Some(ScaleDtype::UE8M0) {
        return Err(PerRowRefusal::BlockScalesNotPowersOfTwo);
    }
    let (rows, blocks) = rows_of(block_scales.shape());
    if !blocks.is_multiple_of(SCALES_PER_WORD) {
        return Err(PerRowRefusal::BlocksNotWholeWords);
    }
    assert!(
        is_contiguous(block_scales),
        "factor_block_scales_by_row: the block scales should lie in dense rows, read as words"
    );
    let words = TensorHandle::new_contiguous(
        [rows, blocks / SCALES_PER_WORD],
        block_scales.handle.clone(),
        ElemType::UInt(UIntKind::U32),
    );
    // Flat, so no runtime pads its rows.
    let offsets = TensorHandle::empty(
        client,
        [rows * blocks / SCALES_PER_WORD],
        ElemType::UInt(UIntKind::U32),
    );
    let row_scales = TensorHandle::empty(client, [rows], ElemType::Float(FloatKind::F32));
    let lossy_blocks = TensorHandle::zeros(client, [1], ElemType::UInt(UIntKind::U32));
    let lossy_past = exact_offsets(scheme.value, decoded_into);
    let address_type = words.required_address_type();
    // SAFETY: one cube a row of scales, which reads and writes that row alone and adds into the
    // one counter atomically; a cube past the rows leaves.
    unsafe {
        cube_block_scales_factor_by_row::launch_unchecked(
            client,
            calculate_cube_count_elemwise(client, rows, CubeDim::new_single()),
            CubeDim::new_1d(ROW_UNITS as u32),
            address_type,
            words.into_arg(),
            offsets.clone().into_arg(),
            row_scales.clone().into_arg(),
            lossy_blocks.clone().into_arg().into_buffer_arg(),
            lossy_past,
        )
    };
    Ok(RowFactoredScales {
        block_scales: TensorHandle::new_contiguous(
            [rows, blocks],
            offsets.handle,
            ElemType::UInt(UIntKind::U8),
        ),
        row_scales,
        lossy_blocks,
    })
}

/// A float weight quantized under a block-scaled scheme, its block scales factored by row.
pub struct RowFactoredTensor {
    /// Its codes, `[rows, k / per word]` words, the lowest first.
    pub codes: TensorHandle,
    /// Its scales.
    pub scales: RowFactoredScales,
}

/// `input`, a float `[rows, k]` of `input_elem`, quantized under `scheme`'s values and
/// power-of-two blocks, then its block scales factored by row ([`factor_block_scales_by_row`]),
/// for a decode into `decoded_into`. Each block's `ue8m0` scale is its largest magnitude over
/// the format's largest, rounded up to a power of two, so no value saturates where the OCP MX
/// rule rounds down and saturates the largest; each value is its nearest code under it.
///
/// # Errors
///
/// [`PerRowRefusal::BlockScalesNotPowersOfTwo`] for a scheme whose blocks are no `ue8m0`, and
/// [`PerRowRefusal::BlocksNotWholeWords`] where a block holds no whole words of codes or `k` no
/// whole words of four block scales.
pub fn quantize_row_factored(
    client: &Client,
    input: &TensorHandle,
    input_elem: ElemType,
    scheme: &QuantScheme,
    decoded_into: ElemType,
) -> Result<RowFactoredTensor, PerRowRefusal> {
    let Some(block) = scheme
        .block_scale()
        .filter(|block| block.dtype == ScaleDtype::UE8M0)
        .map(|block| block.size.to_dim_vec(1)[0] as usize)
    else {
        return Err(PerRowRefusal::BlockScalesNotPowersOfTwo);
    };
    let (rows, k) = rows_of(input.shape());
    let per_word = scheme.num_quants();
    let per_scale_word = SCALES_PER_WORD * block;
    if !block.is_multiple_of(per_word) || !k.is_multiple_of(per_scale_word) {
        return Err(PerRowRefusal::BlocksNotWholeWords);
    }
    // Flat, so no runtime pads their rows.
    let codes = TensorHandle::empty(client, [rows * k / per_word], ElemType::UInt(UIntKind::U32));
    let scales = TensorHandle::empty(
        client,
        [rows * k / per_scale_word],
        ElemType::UInt(UIntKind::U32),
    );
    let units = rows * k / per_scale_word;
    let cube_dim = CubeDim::new(client, units);
    let address_type = input.required_address_type();
    // SAFETY: one unit a word of block scales, which reads its four blocks and writes their codes
    // and scales alone; a unit past the words leaves.
    unsafe {
        unit_block_scaled_quantize::launch_unchecked(
            client,
            calculate_cube_count_elemwise(client, units, cube_dim),
            cube_dim,
            address_type,
            input.clone().into_arg(),
            codes.clone().into_arg(),
            scales.clone().into_arg(),
            *scheme,
            block,
            input_elem,
        )
    };
    let block_scales = TensorHandle::new_contiguous(
        [rows, k / block],
        scales.handle,
        ElemType::UInt(UIntKind::U8),
    );
    Ok(RowFactoredTensor {
        codes: TensorHandle::new_contiguous(
            [rows, k / per_word],
            codes.handle,
            ElemType::UInt(UIntKind::U32),
        ),
        scales: factor_block_scales_by_row(client, scheme, &block_scales, decoded_into)?,
    })
}

/// How many powers of two a block's scale may fall below its row's before its smallest nonzero
/// value, a `value`, is no `element` value: the exponent of `element`'s smallest step less that
/// of `value`'s smallest nonzero. Zero where either is no float this crate knows.
fn exact_offsets(value: QuantValue, element: ElemType) -> u32 {
    let smallest_value = match value {
        QuantValue::E2M1 => -1,
        QuantValue::E4M3 => -9,
        QuantValue::E5M2 => -16,
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => 0,
    };
    let smallest_step = match element {
        ElemType::Float(FloatKind::E4M3) => -9,
        ElemType::Float(FloatKind::E5M2) => -16,
        ElemType::Float(FloatKind::F16) => -24,
        _ => smallest_value,
    };
    (smallest_value - smallest_step).max(0) as u32
}

/// Whether `tensor` lies in dense rows, its strides its shape's.
fn is_contiguous(tensor: &TensorHandle) -> bool {
    let shape = tensor.shape();
    let mut stride = 1;
    (0..shape.rank()).rev().all(|axis| {
        let dense = tensor.strides()[axis] == stride || shape[axis] == 1;
        stride *= shape[axis];
        dense
    })
}

/// A matrix's rows and the extent of each, its innermost dimension.
fn rows_of(shape: &Shape) -> (usize, usize) {
    let inner = shape[shape.rank() - 1];
    (shape.num_elements() / inner.max(1), inner)
}

/// One cube a row: the row's largest magnitude, reduced across the cube, then each unit's words
/// of the scheme's values written under it.
///
/// - `input`: `[rows, k]` floats of `F`, read through its strides.
/// - `words`: `rows · k / per word` flat, the 8-bit values packed a word, the lowest first.
/// - `scales`: `[rows]`, each row's `f32` scale.
#[cube(launch_unchecked, address_type = "dynamic")]
fn cube_fp8_per_row_quantize<F: Float>(
    input: &Tensor<F>,
    words: &mut Tensor<u32>,
    scales: &mut Tensor<f32>,
    #[comptime] scheme: QuantScheme,
    #[define(F)] _float: ElemType,
) {
    let row = CUBE_POS;
    if row >= scales.len() {
        terminate!();
    }
    let per_word = comptime!(scheme.num_quants());
    let largest_value = comptime!(scheme.value.range().1);
    let word_count = input.shape(1) / per_word;
    let (start, step) = (row * input.stride(0), input.stride(1));
    let mut largest = 0f32;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        #[unroll]
        for j in 0..per_word {
            let value = f32::cast_from(input[start + (w * per_word + j) * step]);
            largest = f32::max(largest, f32::abs(value));
        }
    }
    // A magnitude's bits order as the magnitude does.
    let largest = f32::reinterpret(cube_max(u32::reinterpret(largest)));
    let scale = select(largest > 0f32, largest / largest_value, 1f32);
    if UNIT_POS == 0 {
        scales[row] = scale;
    }
    let size!(N) = per_word;
    for w in range_stepped(UNIT_POS as usize, word_count, CUBE_DIM as usize) {
        let mut word = Vector::<f32, N>::empty();
        #[unroll]
        for j in 0..per_word {
            let value = f32::cast_from(input[start + (w * per_word + j) * step]);
            word.insert(j, value / scale);
        }
        words[row * word_count + w] = pack_q::<f32, N, u32>(word, scheme.value);
    }
}

/// One cube a row of block scales: the row's largest, reduced across the cube, then each word of
/// four offsets below it. Unit zero writes the row's scale and adds the row's lossy blocks into
/// the counter.
///
/// - `scales`: `[rows, k / block / 4]`, the `ue8m0` block scales four a word, contiguous.
/// - `offsets`: `rows · k / block / 4` flat, each block's offset below its row's largest, a
///   `ue8m0`.
/// - `row_scales`: `[rows]`, each row's largest block scale as an `f32`.
/// - `lossy`: one counter, the blocks more than `lossy_past` powers of two below their row's.
/// - `lossy_past`: how far below its row's a block's scale decodes exactly.
#[cube(launch_unchecked, address_type = "dynamic")]
fn cube_block_scales_factor_by_row(
    scales: &Tensor<u32>,
    offsets: &mut Tensor<u32>,
    row_scales: &mut Tensor<f32>,
    lossy: &[Atomic<u32>],
    lossy_past: u32,
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
            lossy_here += u32::cast_from(below > lossy_past);
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

/// A unit a word of four block scales: each block's scale, then its codes, the scheme's count a
/// word.
///
/// - `input`: `[rows, k]` floats of `F`, read through its strides.
/// - `codes`: `rows · k / per word` flat, the codes packed a word, the lowest first.
/// - `scales`: `rows · k / block / 4` flat, the `ue8m0` block scales four a word.
#[cube(launch_unchecked, address_type = "dynamic")]
fn unit_block_scaled_quantize<F: Float>(
    input: &Tensor<F>,
    codes: &mut Tensor<u32>,
    scales: &mut Tensor<u32>,
    #[comptime] scheme: QuantScheme,
    #[comptime] block: usize,
    #[define(F)] _float: ElemType,
) {
    if ABSOLUTE_POS >= scales.len() {
        terminate!();
    }
    let per_word = comptime!(scheme.num_quants());
    let largest_value = comptime!(scheme.value.range().1);
    let per_row = input.shape(1) / (SCALES_PER_WORD * block);
    let (row, word) = (ABSOLUTE_POS / per_row, ABSOLUTE_POS % per_row);
    let first = word * SCALES_PER_WORD * block;
    let (start, step) = (row * input.stride(0), input.stride(1));
    let flat = ABSOLUTE_POS * SCALES_PER_WORD * block;
    let size!(N) = per_word;
    let mut scale_word = 0u32;
    #[unroll]
    for b in 0..SCALES_PER_WORD {
        let column = first + b * block;
        let mut largest = 0f32;
        for j in 0..block {
            let value = f32::cast_from(input[start + (column + j) * step]);
            largest = f32::max(largest, f32::abs(value));
        }
        let byte = ue8m0_at_least(largest / largest_value);
        scale_word |= byte << comptime!(BYTE_BITS * b as u32);
        // A zero block's codes are zero whatever its scale.
        let inverse = select(byte == 0, 0f32, power_of_two(2 * EXPONENT_BIAS - byte));
        for w in 0..block / per_word {
            let mut values = Vector::<f32, N>::empty();
            #[unroll]
            for j in 0..per_word {
                let at = column + w * per_word + j;
                values.insert(j, f32::cast_from(input[start + at * step]) * inverse);
            }
            let code_word = (flat + b * block) / per_word + w;
            codes[code_word] = pack_q::<f32, N, u32>(values, scheme.value);
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
