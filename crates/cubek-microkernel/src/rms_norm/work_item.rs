use crate::rms_norm::{GateOrder, RmsNormOperand};
use crate::row_store::RowStore;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;

/// A row of this kind as its microkernels see it: the element the operands are read in and the
/// one the normalized row is written at, the dim the row spans, which every owner holds whole, and
/// how the row is stored, which decides the runs an owner may split it into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNormWorkItem {
    /// The element of the input and the weight.
    pub element: ElemType,
    /// The element the normalized row is written at: `element`, or `f32`.
    pub output_element: ElemType,
    /// The normalized dim: what the mean divides by.
    pub dim: usize,
    /// How the rows are stored, which decides the runs an owner may write.
    pub store: RowStore,
    /// Whether the norm is gated, and where the gate enters.
    pub gate: Option<GateOrder>,
}

impl RmsNormWorkItem {
    /// The element `operand` is stored in.
    pub fn element_of(&self, operand: RmsNormOperand) -> ElemType {
        match operand {
            RmsNormOperand::Out => self.output_element,
            RmsNormOperand::Input | RmsNormOperand::Weight | RmsNormOperand::Gate => self.element,
        }
    }

    /// The element a row is held in between its two passes: `f32` where the gate is applied before
    /// the norm, since a gated row is not an element's value, else the input's.
    pub fn held_element(&self) -> ElemType {
        match self.gate {
            Some(GateOrder::GateThenNorm) => ElemType::Float(FloatKind::F32),
            None => self.element,
        }
    }

    /// The width every operand binds at on `hw`: what the operands' one requirement, read along
    /// the whole row ([`RmsNormOperand::require_along_run`]), settles at the narrowest widest read
    /// of their elements. The operands are bound as dense rows starting on a whole read, so
    /// nothing else narrows it; a microkernel sizes its registers at it, and a routine declines a
    /// blueprint that binds another.
    pub fn width(&self, hw: &Hardware) -> usize {
        let mut vectorization = Vectorization::new();
        RmsNormOperand::require_along_run(self.dim, self.gate, &mut vectorization);
        vectorization.width(RmsNormOperand::Input, |operand| {
            hw.vector_ceiling(self.element_of(operand))
        })
    }

    /// Lines of the row at [`width`](Self::width).
    pub fn lines(&self, hw: &Hardware) -> usize {
        self.dim.div_ceil(self.width(hw)).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rms_norm::HeldRowShape;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn item(gate: Option<GateOrder>) -> RmsNormWorkItem {
        RmsNormWorkItem {
            element: F16,
            output_element: F16,
            dim: 2048,
            store: RowStore::Plain,
            gate,
        }
    }

    /// A row gated before the norm is held in `f32`, which doubles the registers an `f16` row
    /// takes.
    #[test]
    fn a_row_gated_before_the_norm_is_held_in_f32() {
        let shape = HeldRowShape {
            lines: 256,
            units: 32,
        };
        let registers = |gate| {
            let item = item(gate);
            shape.work(8, item.held_element()).registers
        };
        assert_eq!(item(None).held_element(), F16);
        assert_eq!(
            registers(Some(GateOrder::GateThenNorm)) - 1,
            2 * (registers(None) - 1)
        );
    }
}
