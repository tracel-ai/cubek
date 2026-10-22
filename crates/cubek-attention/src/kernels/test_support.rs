//! The devices and problems the family's host-side tests run against.

use crate::kernels::tune::key::AttentionKey;
use crate::kernels::{AttentionRoutineProblem, AttentionVisibility, ScoreScale};
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubek_std::Hardware;
use cubek_std::devices;

/// `hw` offering the square `edge`-wide instruction for `f16`, `bf16` and `f32` operands summed in
/// `f32`.
pub(crate) fn offering(hw: Hardware, edge: u32) -> Hardware {
    let square = |element| MmaConfig {
        a_type: element,
        b_type: element,
        cd_type: ElemType::Float(FloatKind::F32),
        m: edge,
        n: edge,
        k: edge,
    };
    devices::offering_cmma(
        hw,
        [
            square(ElemType::Float(FloatKind::F16)),
            square(ElemType::Float(FloatKind::BF16)),
            square(ElemType::Float(FloatKind::F32)),
        ],
    )
}

/// Qwen3-8B's attention at `f16`, 32 heads over 8 of 128: `queries` positions over `keys`, causal
/// past one position.
pub(crate) fn qwen(queries: usize, keys: usize) -> AttentionRoutineProblem {
    let f16 = ElemType::Float(FloatKind::F16);
    let visibility = AttentionVisibility::new(queries > 1, None);
    let key = AttentionKey::new(
        f16,
        32,
        8,
        128,
        queries,
        keys,
        1,
        visibility,
        ScoreScale::standard(128),
        None,
    );
    AttentionRoutineProblem::try_from(&key).expect("an even group")
}
