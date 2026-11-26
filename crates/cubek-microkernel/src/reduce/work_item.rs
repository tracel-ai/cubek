use crate::reduce::{REDUCED, ReduceOperand, ReduceOperation, ReduceWrites};
use cubecl::ir::ElemType;
use cubecl::tile::Dimension;
use cubecl::tile::launch::VectorRequirement;
use cubek_std::microkernel::{ArithmeticInstruction, UnitWork};
#[cfg(test)]
use cubek_std::microkernel::{CubeFootprint, KernelEstimate};

/// A reduced row as its microkernels see it: what it reduces with, what it reads and accumulates,
/// and which dimension its lines run along. Not how long it is: a microkernel reads its row's length off its tile, so
/// whether the length is compiled in is its launch's choice.
///
/// # Assumptions
///
/// - A top-k keeps at least one value (`k > 0`).
/// - The indices are written beside the values (`with_indices`) only where the operation's carrier
///   keeps them ([`ReduceCarrier::carries_indices`](crate::reduce::ReduceCarrier::carries_indices)):
///   a microkernel writes the indices its carrier keeps.
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
    /// What one unit works with: vector arithmetic, and registers for its carrier's lines beside
    /// two of its own.
    pub fn unit_work(&self) -> UnitWork {
        UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: self.operation.carrier().lines() as u32 + 2,
        }
    }

    /// What a unit that merges stored partials works with: registers for the merged carrier's lines and
    /// one partial's, beside two of its own.
    pub fn merge_work(&self) -> UnitWork {
        UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 2 * self.operation.carrier().lines() as u32 + 2,
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

impl ReduceWorkItem {
    /// Whether its carrier keeps the indices it writes, or it writes none.
    pub(crate) fn has_indices_to_write(&self) -> bool {
        self.writes() == ReduceWrites::Values || self.operation.carrier().carries_indices()
    }

    /// Whether it is rightly handed an output of indices (`handed`): exactly where it writes them
    /// beside the values.
    pub(crate) fn takes_handed_indices(&self, handed: bool) -> bool {
        handed == (self.writes() == ReduceWrites::Both)
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

/// An estimate nothing has been added to.
#[cfg(test)]
pub(crate) fn empty_estimate() -> KernelEstimate {
    KernelEstimate {
        cubes: None,
        cube: CubeFootprint {
            units: 1,
            shared_memory: 0,
        },
        demand: None,
        staging: None,
        work: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::BATCH;
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

    #[test]
    fn across_the_axis_an_input_line_is_an_output_line() {
        assert_eq!(
            item(ReduceOperation::Sum, BATCH, false).requirement(),
            VectorRequirement::Together {
                operand: ReduceOperand::Input,
                with: ReduceOperand::Out,
            }
        );
    }

    /// A unit holds its carrier's lines, beside two registers of its own.
    #[test]
    fn a_unit_holds_its_carriers_lines_and_two() {
        const KEPT_COUNT: u32 = 3;
        let work = item(
            ReduceOperation::ArgTopK {
                k: KEPT_COUNT as usize,
            },
            REDUCED,
            false,
        )
        .unit_work();
        assert_eq!(work.registers, 2 * KEPT_COUNT + 2);
    }

    #[test]
    fn a_merge_holds_two_carriers_lines_and_two() {
        const KEPT_COUNT: u32 = 3;
        let work = item(
            ReduceOperation::TopK {
                k: KEPT_COUNT as usize,
            },
            REDUCED,
            false,
        )
        .merge_work();
        assert_eq!(work.registers, 2 * KEPT_COUNT + 2);
    }

    #[test]
    fn a_unit_computes_on_vectors() {
        let work = item(ReduceOperation::Sum, REDUCED, false).unit_work();
        assert_eq!(work.instruction, ArithmeticInstruction::Vector);
    }

    #[test]
    fn a_sum_has_no_indices_to_write_beside_its_values() {
        assert!(!item(ReduceOperation::Sum, REDUCED, true).has_indices_to_write());
    }

    #[test]
    fn an_arg_max_has_the_indices_it_writes() {
        assert!(item(ReduceOperation::ArgMax, REDUCED, false).has_indices_to_write());
    }

    #[test]
    fn an_item_that_writes_both_takes_indices() {
        assert!(item(ReduceOperation::ArgMax, REDUCED, true).takes_handed_indices(true));
    }

    #[test]
    fn an_item_that_writes_one_half_refuses_indices() {
        assert!(!item(ReduceOperation::ArgMax, REDUCED, false).takes_handed_indices(true));
    }

    #[test]
    fn indices_beside_the_values_write_both() {
        assert_eq!(
            item(ReduceOperation::ArgMax, REDUCED, true).writes(),
            ReduceWrites::Both
        );
    }

    #[test]
    fn without_indices_beside_them_the_operation_writes_its_own() {
        assert_eq!(
            item(ReduceOperation::ArgMax, REDUCED, false).writes(),
            ReduceWrites::Indices
        );
    }
}
