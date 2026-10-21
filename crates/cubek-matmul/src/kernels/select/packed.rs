//! What the selection ([`super::selector::select`]) asks of a packed weight: the
//! schemes the kernels stream ([`matmul_supported`]), and the line between the packed weight's
//! few-rows and many-rows tables ([`MatmulTable::packed`](super::selector::MatmulTable::packed)),
//! [`QUANTIZED_FEW_ROWS`], a **rule** rather than a measurement, for the reason the float bound
//! is one.

use crate::stored::Operand;
use cubecl::quant::scheme::{QuantScheme, QuantValue, ScaleDtype};
use cubek_std::Hardware;
use cubek_std::NVFP4_BLOCK;
use cubek_std::matmul_axes::{K, N};

/// The weight a scheme describes, read before any tensor is in hand.
///
/// A rule runs ahead of the launch, so the strides that decide the axis
/// everywhere else are not here and the scheme's own claim answers.
fn scheme_operand(scheme: &QuantScheme) -> Option<Operand> {
    Operand::of_scheme(scheme, Operand::axis_of(scheme, [K, N])?, [K, N])
}

/// The most rows a packed product goes to the few-rows table with. Past it the staged routines win on
/// arithmetic, and measuring a gemv at prefill widths would cost minutes of first-touch tuning for
/// a candidate that cannot win there.
pub const QUANTIZED_FEW_ROWS: usize = 32;

/// Whether a packed product of `rows` rows goes to the few-rows table: up to [`QUANTIZED_FEW_ROWS`],
/// no rows included, the packed candidates' own admission answering every width below the bound.
pub fn quantized_few_rows(rows: usize) -> bool {
    rows <= QUANTIZED_FEW_ROWS
}

/// Whether the packed kernels stream `scheme`: the packed int formats
/// (`q4f`/`q4s`/`q8f`/`q8s`) or the fp4 one (`e2m1`), in either packed-u32
/// form — `PackedU32(0)` with blocks along the output dimension (burn's
/// artifact form) or `PackedU32(1)` with blocks along the reduction (the
/// col-requantized serving form) — with block-level scales running the packed
/// direction and covering whole `u32`s (see
/// [`Operand::of_scheme`](crate::stored::Operand::of_scheme)), at any block-scale
/// dtype, and under a per-tensor second level or not.
///
/// It takes the device because streaming a packed field means *decoding* it,
/// and whether a decode runs is the device's answer, not the scheme's: asking
/// the scheme alone once said `true` on a device whose kernel failed at codegen
/// on a worker thread and left its output as the allocator had it. Every
/// decode the engine has today is software — `e2m1` included, unpacked as
/// every other read of it is — so the device currently declines nothing, and
/// the parameter is where the next format that needs a named type asks.
///
/// A scheme outside this set still goes through the family's entry — dequantized
/// first, or on the framework's quantized matmul — it just never reaches one
/// of this crate's kernels packed. This is the gate that decides which, so
/// everything downstream may take the form, and the decode, for granted.
pub fn matmul_supported(_hw: &Hardware, scheme: &QuantScheme) -> bool {
    let value = matches!(
        scheme.value,
        QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q8F
            | QuantValue::Q8S
            // The fp4 formats' value type: a code rather than a small integer, decoded in
            // software wherever it is read. NVFP4 and MXFP4 differ only in their scale
            // levels, which the packing answers.
            | QuantValue::E2M1
    );
    value && scheme_operand(scheme).is_some()
}

/// Values of `k` one word of an NVFP4 weight's scales covers: four one-byte scales, a block each.
pub const NVFP4_SCALE_WORD_SPAN: usize = NVFP4_BLOCK * 4;

/// Whether a `[k, n]` weight under `scheme` reads as an NVFP4 operand's rows: `e2m1` values under
/// a `ue4m3` scale every [`NVFP4_BLOCK`] of `k` and one `f32` over the whole operand, its words
/// running along `k`, `k` a whole number of words of scales. What a rule asks before a tensor is
/// in hand.
pub fn nvfp4_admits(scheme: &QuantScheme, k: usize) -> bool {
    let format = scheme.value == QuantValue::E2M1
        && scheme.block_scale().map(|block| block.dtype) == Some(ScaleDtype::UE4M3)
        && scheme.tensor_scale() == Some(ScaleDtype::F32);
    let along_k = Operand::axis_of(scheme, [K, N]) == Some(K)
        && Operand::of_scheme(scheme, K, [K, N])
            .is_some_and(|operand| operand.scale(K) == Some(NVFP4_BLOCK));
    format && along_k && k.is_multiple_of(NVFP4_SCALE_WORD_SPAN)
}
