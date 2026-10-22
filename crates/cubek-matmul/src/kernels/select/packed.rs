//! What the selection ([`super::selector::select`]) asks of a packed weight: the
//! schemes the kernels stream ([`matmul_supported`]), and the line between the packed weight's
//! few-rows and many-rows tables ([`MatmulTable::packed`](super::selector::MatmulTable::packed)),
//! [`QUANTIZED_FEW_ROWS`], a **rule** rather than a measurement, for the reason the float bound
//! is one.

use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_microkernel::matmul::stored::Operand;
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
/// arithmetic, and measuring a routine that does not stage at prefill widths would cost minutes of first-touch tuning for
/// a candidate that cannot win there.
pub const QUANTIZED_FEW_ROWS: usize = 32;

/// Whether a packed product of `rows` rows goes to the few-rows table: up to [`QUANTIZED_FEW_ROWS`],
/// no rows included, the packed candidates' own admission answering every width below the bound.
pub fn quantized_few_rows(rows: usize) -> bool {
    rows <= QUANTIZED_FEW_ROWS
}

/// Whether the packed kernels stream `scheme`: the packed int formats
/// (`q4f`/`q4s`/`q8f`/`q8s`), the fp4 one (`e2m1`) or the fp8 ones (`e4m3`/`e5m2`), in
/// either packed-u32 form — `PackedU32(0)` with blocks along the output dimension (the
/// framework's artifact form) or `PackedU32(1)` with blocks along the reduction (the
/// col-requantized serving form) — with block-level scales running the packed
/// direction and covering whole `u32`s (see
/// [`Operand::of_scheme`](cubek_microkernel::matmul::stored::Operand::of_scheme)), at any block-scale
/// dtype, and under a per-tensor second level or not.
///
/// It takes the device because streaming a packed field means *decoding* it,
/// and whether a decode runs is the device's answer, not the scheme's: asking
/// the scheme alone once said `true` on a device whose kernel failed at codegen
/// on a worker thread and left its output as the allocator had it. Every
/// decode the kernels have is software — `e2m1` included, unpacked as
/// every other read of it is — so the device currently declines nothing, and
/// the parameter is where the next format that needs a named type asks.
///
/// A scheme outside this set still goes through the family's entry — dequantized
/// first, or on the framework's quantized matmul — it just never reaches one
/// of the tiled kernels packed. This is the gate that decides which, so
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
            // The fp8 formats' value types, MXFP8's among them: a byte a code, four a word,
            // decoded in software as `e2m1` is.
            | QuantValue::E4M3
            | QuantValue::E5M2
    );
    value && scheme_operand(scheme).is_some()
}

/// Values of `k` one word of an NVFP4 weight's scales covers: four one-byte scales, a block each.
pub const NVFP4_SCALE_WORD_SPAN: usize = NVFP4_BLOCK * 4;

/// The scheme an activation is quantized under for the 4-bit tensor cores: `e2m1` values along
/// its last axis, eight to a word, under a `ue4m3` scale every [`NVFP4_BLOCK`] and one `f32` over
/// the whole of it.
pub fn nvfp4_activation_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([NVFP4_BLOCK as u8], ScaleDtype::UE4M3)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
}
