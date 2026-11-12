use super::UnitBlock;
use crate::kernels::tiled::{ConvProblem, ConvRoles};
use cubecl::ir::ElemType;
use cubek_microkernel::matmul::MatmulLayout;
use cubek_std::Hardware;

/// How a unit's register block reads the input, which the axis the input runs along decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockForm {
    /// The input runs down the contraction: a cell holds one vector of partial sums, and a step
    /// reads one vector for each row and each column.
    Dot,
    /// The input runs down the block's rows: a step reads a vector of the rows and one value for
    /// each column, and each column of cells is a vector of the rows.
    Outer,
}

impl BlockForm {
    /// Blocks along an axis the search tries at most.
    const MOST_ALONG: usize = 4;

    /// Vector registers left to addressing and the compiler's temporaries.
    const RESERVED_REGISTERS: usize = 2;

    /// The form `roles` give a block.
    pub fn of(roles: &ConvRoles) -> Self {
        match roles.layouts().lhs {
            MatmulLayout::Col => Self::Outer,
            MatmulLayout::Row | MatmulLayout::Strided => Self::Dot,
        }
    }

    /// The rows and columns `block` asks of a block of this form on `hw`, summing in `sum`.
    pub fn size(self, block: UnitBlock, hw: &Hardware, sum: ElemType) -> (usize, usize) {
        match block {
            UnitBlock::Largest => self.largest_block(hw, sum),
            UnitBlock::OneCell => (1, 1),
        }
    }

    /// The rows and columns of the largest block of this form summing in `sum` that fits the
    /// registers the device keeps live less the reserved ones.
    pub fn largest_block(self, hw: &Hardware, sum: ElemType) -> (usize, usize) {
        let registers = hw.vector_registers(sum);
        let budget = registers.count().saturating_sub(Self::RESERVED_REGISTERS);
        let along = |most: usize| (1..=most).collect::<Vec<_>>();
        let powers = |most: usize| {
            (0..)
                .map(|power| 1_usize << power)
                .take_while(|&size| size <= most)
                .collect::<Vec<_>>()
        };
        let (rows, columns) = match self {
            Self::Dot => (along(Self::MOST_ALONG), along(Self::MOST_ALONG)),
            Self::Outer => (powers(Self::MOST_ALONG), powers(budget)),
        };
        let cost = |rows: usize, columns: usize| match self {
            Self::Dot => rows * columns + rows + columns,
            Self::Outer => {
                let row_registers = registers.registers_for(rows);
                columns * row_registers + row_registers + registers.registers_for(columns)
            }
        };
        rows.iter()
            .flat_map(|&rows| columns.iter().map(move |&columns| (rows, columns)))
            .filter(|&(rows, columns)| cost(rows, columns) <= budget)
            .max_by_key(|&(rows, columns)| (rows * columns, rows))
            .unwrap_or((1, 1))
    }
}

/// What sizes a block's depth, the contraction one step reads, which turns on whether the device
/// states the vector registers a unit keeps live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDepth {
    /// A device stating them, as a CPU runtime does: as many partial sums as one register holds. A
    /// dot block's cell is then the one register [`BlockForm::largest_block`] counts it as; an
    /// outer block's cell is a vector of its rows, so there the depth is only what a step reads.
    OneRegister,
    /// A device stating none, as a GPU, whose partials are scalars: the widest read of the input.
    WidestRead,
}

impl BlockDepth {
    /// The rule `hw` sizes a block's depth by.
    pub fn of(hw: &Hardware) -> Self {
        match hw.properties.hardware.vector_register_count {
            Some(_) => Self::OneRegister,
            None => Self::WidestRead,
        }
    }

    /// The depth of a block of `problem` on `hw`.
    pub fn size(self, hw: &Hardware, problem: &ConvProblem) -> usize {
        match self {
            Self::OneRegister => hw.register_values(problem.sum),
            Self::WidestRead => hw.vector_ceiling(problem.elements.lhs),
        }
    }
}
