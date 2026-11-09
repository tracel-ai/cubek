use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of `out = silu(gate) ⊙ up`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwigluOperand {
    /// `[r, h, d]`: the fused rows, the gate half at `h = 0` and the up half at `h = 1`.
    Input,
    /// `[r, d]`.
    Out,
}

impl SwigluOperand {
    /// The two operands, in the order this kind names them.
    pub const ALL: [Self; 2] = [Self::Input, Self::Out];

    /// What every microkernel of this kind requires of the operands, each reading and writing a
    /// run of `elements` of a half one line of each at a time: the two share one width, and it
    /// divides the run.
    pub fn require_along_run(elements: usize, vectorization: &mut Vectorization<Self>) {
        for operand in Self::ALL {
            vectorization.require(VectorRequirement::Access { operand, elements });
        }
        vectorization.require(VectorRequirement::Together {
            operand: Self::Input,
            with: Self::Out,
        });
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwigluOperands<T> {
    pub input: T,
    pub out: T,
}

impl<T: Copy> SwigluOperands<T> {
    /// The value `operand` holds.
    pub fn get(&self, operand: SwigluOperand) -> T {
        match operand {
            SwigluOperand::Input => self.input,
            SwigluOperand::Out => self.out,
        }
    }
}
