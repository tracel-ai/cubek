use crate::reduce::{ReduceOperand, ReduceOperation, ReduceWrites};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// How a reduction's lines lie against the axis it reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceLayout {
    /// The reduced axis is the contiguous one: a line is `width` consecutive values of one row,
    /// whose lanes reduce to one value at the end.
    Along,
    /// Another axis is the contiguous one: a line is one position of `width` rows side by side,
    /// each lane its own row's, and nothing reduces across lanes.
    Across,
}

/// A reduced row as its microkernels see it: what it reduces with, what it reads and accumulates,
/// and how its lines lie. Not how long it is: a microkernel reads its row's length off its tile, so
/// whether the length is compiled in is its launch's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceWorkItem {
    pub operation: ReduceOperation,
    /// The element the input is stored in.
    pub input: ElemType,
    /// The element the values accumulate in.
    pub accumulator: ElemType,
    pub layout: ReduceLayout,
    /// Whether the indices are written beside the values, to a second output laid out as the
    /// first: an argmax's or a top-k's both halves from one launch.
    pub with_indices: bool,
}

impl ReduceWorkItem {
    /// What the owner writes: both halves where the indices go beside the values, what the
    /// operation writes otherwise.
    pub fn writes(&self) -> ReduceWrites {
        match self.with_indices {
            true => ReduceWrites::Both,
            false => self.operation.writes(),
        }
    }

    /// What every microkernel of this kind asks of the operands: along the reduced axis, the
    /// output takes one cell per row (`k` for a top-k); across it, a line of the input is a line
    /// of the output, one row a lane. That a line never runs past a row is the launch's to
    /// require, which knows the rows' length.
    pub fn require(&self, vectorization: &mut Vectorization<ReduceOperand>) {
        match self.layout {
            ReduceLayout::Along => {
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
