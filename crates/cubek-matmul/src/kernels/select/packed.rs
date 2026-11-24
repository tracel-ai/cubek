//! What the selection ([`super::selector::select`]) asks of packed inputs: whether the kernels
//! stream one's scheme ([`matmul_supported`]), and whether a block-scaled instruction reads both
//! (`block_scaled_instruction_reads`).

use super::problem::OperandKind;
use crate::kernels::CubeStagedMatmulRoutine;
use cubecl::quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::NVFP4_BLOCK;
use cubek_std::layout::stored::{Operand, Packing};
use cubek_std::matmul_dimensions::K;

/// Whether the packed kernels stream `input` packed under `scheme`: the packed int formats
/// (`q4f`/`q4s`/`q8f`/`q8s`), the fp4 one (`e2m1`) or the fp8 ones (`e4m3`/`e5m2`), in
/// either packed-u32 form — `PackedU32(0)` with blocks along the output dimension (the
/// framework's artifact form) or `PackedU32(1)` with blocks along the reduction (the
/// col-requantized serving form) — with block-level scales running the packed
/// direction and covering whole `u32`s (see
/// [`Operand::of_scheme`](cubek_std::layout::stored::Operand::of_scheme)), at any block-scale
/// dtype, and under a per-tensor second level or not. The values must be numbers
/// ([`QuantMode::Symmetric`]): an index into a table ([`QuantMode::Lookup`]) carries its table
/// beside it, which no packed input here binds, so its codes would be read as the integers they
/// are stored as.
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
pub fn matmul_supported(_hw: &Hardware, scheme: &QuantScheme, input: MatmulOperand) -> bool {
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
    let numbers = scheme.mode == QuantMode::Symmetric;
    // A rule runs ahead of the launch, so the strides that decide the dimension everywhere else are
    // not here and the scheme's own claim answers.
    let dimensions = input.dimensions();
    let laid = Operand::dimension_of(scheme, dimensions)
        .and_then(|dimension| Operand::of_scheme(scheme, dimension, dimensions))
        .is_some();
    value && numbers && laid
}

/// Whether a block-scaled instruction of `hw` reads both inputs of `operands` as they are stored,
/// their words along `k` ([`CubeStagedMatmulRoutine::block_scaled_step`]).
pub(crate) fn block_scaled_instruction_reads(
    hw: &Hardware,
    operands: &MatmulOperands<OperandKind>,
) -> bool {
    let inputs = operands.map(|operand, kind| match kind {
        OperandKind::Quantized(scheme) => packed_along_k(operand, scheme),
        OperandKind::Float => None,
    });
    CubeStagedMatmulRoutine::block_scaled_step(hw, &inputs).is_some()
}

/// How `operand` is packed under `scheme`, where its words run along `k`.
fn packed_along_k(operand: MatmulOperand, scheme: QuantScheme) -> Option<Packing> {
    let dimensions = operand.dimensions();
    let along = Operand::dimension_of(&scheme, dimensions)?;
    let operand = Operand::of_scheme(&scheme, K, dimensions)?;
    (along == K).then_some(Packing { operand, scheme })
}

/// Values of `k` one word of an NVFP4 input's scales covers: four one-byte scales, a block each.
pub const NVFP4_SCALE_WORD_SPAN: usize = NVFP4_BLOCK * 4;

/// NVFP4 over a matrix stored with its contraction innermost: `e2m1` values along its last axis,
/// eight to a word, under a `ue4m3` scale every [`NVFP4_BLOCK`] and one `f32` over the whole of
/// it.
pub fn nvfp4_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([NVFP4_BLOCK as u8], ScaleDtype::UE4M3)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
}
