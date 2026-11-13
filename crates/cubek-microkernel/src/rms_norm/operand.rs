use crate::rms_norm::GateOrder;
use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of `out = x / sqrt(mean(x²) + eps) · weight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RmsNormOperand {
    /// `[r, d]`: the rows normalized.
    Input,
    /// `[d]`: the scale every row is multiplied by once normalized.
    Weight,
    /// `[r, d]`: the gate, where the norm is gated.
    Gate,
    /// `[r, d]`.
    Out,
}

impl RmsNormOperand {
    /// The operands a norm binds, in the order this kind names them: the gate only where the norm
    /// is gated.
    pub fn of(gate: Option<GateOrder>) -> &'static [Self] {
        match gate {
            None => &[Self::Input, Self::Weight, Self::Out],
            Some(_) => &[Self::Input, Self::Weight, Self::Gate, Self::Out],
        }
    }

    /// What every microkernel of this kind requires of the operands, each reading and writing a
    /// run of `elements` of the row one line of each at a time: they share one width, and it
    /// divides the run.
    pub fn require_along_run(
        elements: usize,
        gate: Option<GateOrder>,
        vectorization: &mut Vectorization<Self>,
    ) {
        for &operand in Self::of(gate) {
            vectorization.require(VectorRequirement::Access { operand, elements });
            if operand != Self::Input {
                vectorization.require(VectorRequirement::Together {
                    operand: Self::Input,
                    with: operand,
                });
            }
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNormOperands<T> {
    pub input: T,
    pub weight: T,
    pub gate: T,
    pub out: T,
}

impl<T: Copy> RmsNormOperands<T> {
    /// The value `operand` holds.
    pub fn get(&self, operand: RmsNormOperand) -> T {
        match operand {
            RmsNormOperand::Input => self.input,
            RmsNormOperand::Weight => self.weight,
            RmsNormOperand::Gate => self.gate,
            RmsNormOperand::Out => self.out,
        }
    }
}
