//! The four operands of the decode gemv, and the axes they are laid out over.
//!
//! Four, not three: the scales are a tensor of their own, so they are an operand of their own.
//! What makes one of their values cover a block of the contraction is that they omit `KI` — a
//! fact about which axes the operand distinguishes, with nothing dividing anything.

use cubecl::ir::ElemType;
use cubek_tile::{Axis, Operand, OperandSet};

use crate::tiled::{M, N};

/// Which block of the contraction, and where inside it. Together they are `K`.
///
/// Numbered past [`batch_axis`](crate::tiled::batch_axis)'s first slots because a gemv states
/// no batch axes: the labels only have to be distinct within the space that uses them.
pub(super) const KB: Axis = Axis(16);
pub(super) const KI: Axis = Axis(17);

/// The weight over `[M, KB, KI]`, the activation over `[N, KB, KI]`, the scales over `[M, KB]`,
/// the output over `[M, N]`.
///
/// The weight is the **lhs**: its physical `[d_out, d_in]` buffer binds as it lies, so `M` is
/// the weight's output dimension and the contraction runs along the buffer's contiguous
/// direction. The activation is read `K`-innermost for the same reason the register contraction
/// wants it: a step folds a whole line only where the rhs's innermost axis is the contracted one.
pub(super) struct QuantGemvOperands {
    pub w: Operand,
    pub x: Operand,
    pub scales: Operand,
    pub out: Operand,
}

impl QuantGemvOperands {
    /// `served` is the element the packed words decode to and the contraction runs in.
    pub fn new(served: ElemType, x: ElemType, scales: ElemType, out: ElemType) -> Self {
        QuantGemvOperands {
            w: Operand::new(&[M, KB, KI], served),
            x: Operand::new(&[N, KB, KI], x),
            scales: Operand::new(&[M, KB], scales),
            out: Operand::new(&[M, N], out),
        }
    }
}

impl OperandSet for QuantGemvOperands {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.w, &mut self.x, &mut self.scales, &mut self.out].into_iter()
    }
}
