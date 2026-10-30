use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of `out = x / sqrt(mean(x²) + eps) · weight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RmsNormOperand {
    /// `[r, d]`: the rows normalized.
    Input,
    /// `[d]`: the scale every row is multiplied by once normalized.
    Weight,
    /// `[r, d]`.
    Out,
}

impl RmsNormOperand {
    /// The three operands, in the order this kind names them.
    pub const ALL: [Self; 3] = [Self::Input, Self::Weight, Self::Out];

    /// What every microkernel of this kind requires of the operands, each reading and writing a
    /// run of `elements` of the row one line of each at a time: the three share one width, and it
    /// divides the run.
    pub fn require_along_run(elements: usize, vectorization: &mut Vectorization<Self>) {
        for operand in Self::ALL {
            vectorization.require(VectorRequirement::Access { operand, elements });
        }
        for with in [Self::Weight, Self::Out] {
            vectorization.require(VectorRequirement::Together {
                operand: Self::Input,
                with,
            });
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNormOperands<T> {
    pub input: T,
    pub weight: T,
    pub out: T,
}

impl<T: Copy> RmsNormOperands<T> {
    /// The value `operand` holds.
    pub fn get(&self, operand: RmsNormOperand) -> T {
        match operand {
            RmsNormOperand::Input => self.input,
            RmsNormOperand::Weight => self.weight,
            RmsNormOperand::Out => self.out,
        }
    }
}
