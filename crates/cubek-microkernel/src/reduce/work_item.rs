use crate::reduce::{ReduceOperand, ReduceOperation};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// How a reduction's lines lie against the axis it reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceLayout {
    /// The reduced axis is the contiguous one: a line is `width` consecutive values of one row,
    /// whose lanes fold into one value at the end.
    Along,
    /// Another axis is the contiguous one: a line is one position of `width` rows side by side,
    /// each lane its own row's, and nothing folds across lanes.
    Across,
}

/// A reduced row as its microkernels see it: what it reduces with, what it reads and folds in,
/// and how its lines lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceWorkItem {
    pub operation: ReduceOperation,
    /// The element the input is stored in.
    pub input: ElemType,
    /// The element the values fold in.
    pub accumulator: ElemType,
    /// Values along the reduced axis.
    pub reduced: usize,
    pub layout: ReduceLayout,
}

impl ReduceWorkItem {
    /// What every microkernel of this kind asks of the operands: along the reduced axis, the
    /// input reads stretches of the row and the output takes one cell per row (`k` for a top-k);
    /// across it, a line of the input is a line of the output, one row a lane.
    pub fn require(&self, vectorization: &mut Vectorization<ReduceOperand>) {
        match self.layout {
            ReduceLayout::Along => {
                vectorization.require(VectorRequirement::Access {
                    operand: ReduceOperand::Input,
                    elements: self.reduced,
                });
                vectorization.require(VectorRequirement::Elementwise {
                    operand: ReduceOperand::Out,
                });
            }
            ReduceLayout::Across => {
                vectorization.require(VectorRequirement::Together {
                    operand: ReduceOperand::Input,
                    with: ReduceOperand::Out,
                });
            }
        }
    }
}
