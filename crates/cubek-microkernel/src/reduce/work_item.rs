use crate::reduce::{REDUCED, ReduceOperand, ReduceOperation, ReduceWrites};
use cubecl::ir::ElemType;
use cubecl::tile::Dimension;
use cubecl::tile::launch::VectorRequirement;
use cubek_std::microkernel::{ArithmeticInstruction, UnitWork};

/// A reduced row as its microkernels see it: what it reduces with, what it reads and accumulates,
/// and which dimension its lines run along. Not how long it is: a microkernel reads its row's length off its tile, so
/// whether the length is compiled in is its launch's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceWorkItem {
    pub operation: ReduceOperation,
    /// The element the input is stored in.
    pub input: ElemType,
    /// The element the values accumulate in.
    pub accumulator: ElemType,
    /// The dimension the input's lines run along: [`REDUCED`] where a line holds values of one
    /// row, [`BATCH`](crate::reduce::BATCH) where each lane is its own row's. The host plans by it;
    /// the device reads the same fact off its input's tile ([`Tile::vector_tile`](cubecl::tile::Tile::vector_tile)).
    pub lines_along: Dimension,
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

    /// What every microkernel of this kind asks of the operands: lines along [`REDUCED`], the
    /// output takes one cell per row (`k` for a top-k); lines across it, a line of the input is a
    /// line of the output, one row a lane. That a line never runs past a row is the launch's to
    /// require, which knows the rows' length.
    pub fn requirement(&self) -> VectorRequirement<ReduceOperand> {
        if self.lines_along == REDUCED {
            return VectorRequirement::Elementwise {
                operand: ReduceOperand::Out,
            };
        }
        VectorRequirement::Together {
            operand: ReduceOperand::Input,
            with: ReduceOperand::Out,
        }
    }
}

#[cfg(test)]
impl ReduceWorkItem {
    /// `operation` on f32 values, its lines along the row and no indices beside them: what a
    /// microkernel's Small Tests reduce.
    pub(crate) fn along_f32(operation: ReduceOperation) -> Self {
        let f32 = ElemType::Float(cubecl::ir::FloatKind::F32);
        Self {
            operation,
            input: f32,
            accumulator: f32,
            lines_along: REDUCED,
            with_indices: false,
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
        lines_along: Dimension,
        with_indices: bool,
    ) -> ReduceWorkItem {
        ReduceWorkItem {
            operation,
            input: F32,
            accumulator: F32,
            lines_along,
            with_indices,
        }
    }

    /// Along the reduced dim the output takes one cell a row.
    #[test]
    fn along_the_axis_the_output_takes_a_cell_a_row() {
        assert_eq!(
            item(ReduceOperation::Sum, REDUCED, false).requirement(),
            VectorRequirement::Elementwise {
                operand: ReduceOperand::Out
            }
        );
    }

    /// Across the reduced dim an input line is an output line.
    #[test]
    fn across_the_axis_an_input_line_is_an_output_line() {
        assert_eq!(
            item(ReduceOperation::Sum, crate::reduce::BATCH, false).requirement(),
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
        let work = item(ReduceOperation::TopK { k: K as usize }, REDUCED, false).unit_work();
        assert_eq!(work.registers, 2 * K + 2);
    }

    /// A unit computes on vectors.
    #[test]
    fn a_unit_computes_on_vectors() {
        let work = item(ReduceOperation::Sum, REDUCED, false).unit_work();
        assert_eq!(work.instruction, ArithmeticInstruction::Vector);
    }

    /// With indices beside them, both halves are written.
    #[test]
    fn indices_beside_the_values_write_both() {
        assert_eq!(
            item(ReduceOperation::ArgMax, REDUCED, true).writes(),
            ReduceWrites::Both
        );
    }

    /// Without indices beside them, what the operation writes.
    #[test]
    fn without_indices_beside_them_the_operation_writes_its_own() {
        assert_eq!(
            item(ReduceOperation::ArgMax, REDUCED, false).writes(),
            ReduceWrites::Indices
        );
    }
}
