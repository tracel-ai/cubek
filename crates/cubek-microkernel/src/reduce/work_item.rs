use crate::reduce::{ReduceOperand, ReduceOperation, ReduceWrites};
use cubecl::ir::ElemType;
use cubecl::tile::launch::VectorRequirement;
use cubek_std::microkernel::{ArithmeticInstruction, UnitWork};

/// How a reduction's lines lie against the dimension it reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReduceLayout {
    /// The reduced dimension is the contiguous one: a line is `width` consecutive values of one
    /// row, whose lanes reduce to one value at the end.
    Along,
    /// Another dimension is the contiguous one: a line is one position of `width` rows side by
    /// side, each lane its own row's, and nothing reduces across lanes.
    Across,
}

impl ReduceLayout {
    /// Whether the lanes of a line `lanes` wide hold values of one row, which then reduce across
    /// lanes: along the row, past one lane.
    pub const fn lanes_share_a_row(self, lanes: usize) -> bool {
        matches!(self, ReduceLayout::Along) && lanes > 1
    }
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
    /// What one unit of any reduce microkernel works with: vector arithmetic, and registers for a
    /// value and an index a slot of the row it writes, beside two of its own.
    pub fn unit_work(&self) -> UnitWork {
        let slots = self.operation.written_per_row() as u32;
        UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 2 * slots + 2,
        }
    }

    /// What the owner writes: both halves where the indices go beside the values, what the
    /// operation writes otherwise.
    pub fn writes(&self) -> ReduceWrites {
        match self.with_indices {
            true => ReduceWrites::Both,
            false => self.operation.writes(),
        }
    }

    /// What every microkernel of this kind asks of the operands: along the reduced dimension, the
    /// output takes one cell per row (`k` for a top-k); across it, a line of the input is a line
    /// of the output, one row a lane. That a line never runs past a row is the launch's to
    /// require, which knows the rows' length.
    pub fn requirement(&self) -> VectorRequirement<ReduceOperand> {
        match self.layout {
            ReduceLayout::Along => VectorRequirement::Elementwise {
                operand: ReduceOperand::Out,
            },
            ReduceLayout::Across => VectorRequirement::Together {
                operand: ReduceOperand::Input,
                with: ReduceOperand::Out,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    fn item(
        operation: ReduceOperation,
        layout: ReduceLayout,
        with_indices: bool,
    ) -> ReduceWorkItem {
        ReduceWorkItem {
            operation,
            input: F32,
            accumulator: F32,
            layout,
            with_indices,
        }
    }

    /// Along the row, a line of several lanes holds one row's values; one lane, or a line across
    /// rows, does not.
    #[test]
    fn lanes_share_a_row_only_along_it_past_one_lane() {
        assert!(ReduceLayout::Along.lanes_share_a_row(4));
        assert!(!ReduceLayout::Along.lanes_share_a_row(1));
        assert!(!ReduceLayout::Across.lanes_share_a_row(4));
    }

    /// Along the reduced dim the output takes one cell a row; across it, an input line is an
    /// output line.
    #[test]
    fn the_requirement_follows_the_layout() {
        assert_eq!(
            item(ReduceOperation::Sum, ReduceLayout::Along, false).requirement(),
            VectorRequirement::Elementwise {
                operand: ReduceOperand::Out
            }
        );
        assert_eq!(
            item(ReduceOperation::Sum, ReduceLayout::Across, false).requirement(),
            VectorRequirement::Together {
                operand: ReduceOperand::Input,
                with: ReduceOperand::Out,
            }
        );
    }

    /// A unit holds a value and an index a slot, beside two registers of its own.
    #[test]
    fn a_unit_holds_two_registers_a_slot_and_two() {
        const K: u32 = 3;
        let work = item(
            ReduceOperation::TopK { k: K as usize },
            ReduceLayout::Along,
            false,
        )
        .unit_work();
        assert_eq!(work.registers, 2 * K + 2);
        assert_eq!(work.instruction, ArithmeticInstruction::Vector);
    }

    /// With indices beside them, both halves are written; otherwise what the operation writes.
    #[test]
    fn indices_beside_the_values_write_both() {
        assert_eq!(
            item(ReduceOperation::ArgMax, ReduceLayout::Along, true).writes(),
            ReduceWrites::Both
        );
        assert_eq!(
            item(ReduceOperation::ArgMax, ReduceLayout::Along, false).writes(),
            ReduceWrites::Indices
        );
    }
}
