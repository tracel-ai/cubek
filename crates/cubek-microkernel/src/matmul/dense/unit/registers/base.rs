use crate::matmul::{
    MatmulAxes, MatmulDim, MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands,
    MatmulOrder, MatmulRowsKind, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Instruction, Levels, RegisterBlock};
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};
use cubek_std::{Hardware, UNIT_REGISTERS};

/// A block of the output one unit owns, summed in registers while it walks `K`, as its settings
/// size it. It asks nothing of the hardware, so it is its settings and the order it reads the rhs
/// in.
///
/// It reads the rhs along the axis its buffer runs: a col rhs down `K`, a line of it merged with a
/// line of the lhs into each cell, and a row rhs, or one lined along neither axis, along `N`,
/// lined with the output's cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegistersMicrokernel {
    pub settings: UnitRegistersMicrokernelSettings,
    /// The order each operand is read in.
    pub orders: MatmulOperands<MatmulOrder>,
    /// The caller's axes its levels are stated along.
    pub axes: MatmulAxes,
}

/// The size of the block a unit owns, and how deep one step of its walk reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegistersMicrokernelSettings {
    pub rows: NonZeroUsize,
    pub columns: NonZeroUsize,
    pub depth: NonZeroUsize,
}

impl UnitRegistersMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        rows: NonZeroUsize::new(4).unwrap(),
        columns: NonZeroUsize::new(4).unwrap(),
        depth: NonZeroUsize::new(4).unwrap(),
    };
}

impl core::fmt::Display for UnitRegistersMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "block={}x{}x{}", self.rows, self.columns, self.depth)
    }
}

impl UnitRegistersMicrokernel {
    /// The levels inside one unit's block, which a routine states the rest of the partitioning
    /// around: one step of the block, walked through all of `K`.
    pub fn tiling(&self) -> Levels {
        let [rows, columns, depth] = self.extents();
        let MatmulAxes { m, n, k } = self.axes;
        Levels::leaf(&[(m, rows), (n, columns), (k, depth)]).walk_every(&[k])
    }

    /// 32-bit registers one unit keeps its block in: its accumulator, and one step's rows of the
    /// lhs and columns of the rhs. A lower bound: the compiler adds addressing and loop state.
    pub fn registers_per_unit(&self, item: &MatmulWorkItem) -> u32 {
        let [rows, columns, depth] = self.extents();
        let bytes = rows * columns * item.sum.size()
            + rows * depth * item.elements.lhs.size()
            + depth * columns * item.elements.rhs.size();
        bytes.div_ceil(4) as u32
    }

    /// The instruction a unit's sum opens on: a register block of its accumulator's cells,
    /// unrolled while they fit a unit's registers. A block past them, such as an attention unit's
    /// sum over a 512-wide value dim, walks its cells at runtime.
    pub fn tile_instruction(&self) -> Instruction {
        Instruction::Registers {
            config: RegisterBlock::new(UNIT_REGISTERS as usize),
        }
    }

    /// The extent along `dim` of one step of the block.
    fn extent(&self, dim: MatmulDim) -> usize {
        let [rows, columns, depth] = self.extents();
        match dim {
            MatmulDim::M => rows,
            MatmulDim::N => columns,
            MatmulDim::K => depth,
        }
    }

    /// The block's `rows × columns`, and the `depth` one step reads.
    fn extents(&self) -> [usize; 3] {
        let settings = self.settings;
        [
            settings.rows.get(),
            settings.columns.get(),
            settings.depth.get(),
        ]
    }
}

/// Its block, and the order it reads the rhs in.
impl core::fmt::Display for UnitRegistersMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} rhs={}", self.settings, self.orders.rhs)
    }
}

impl Microkernel for UnitRegistersMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = UnitRegistersMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = ();

    const DECLINE: &'static str = "the unit microkernel asks nothing of the hardware";

    /// The default block, then a smaller one and the two that double it along one axis: a larger
    /// block reuses each read more and holds more registers.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        let block = |rows: usize, columns: usize| Self::Settings {
            rows: NonZeroUsize::new(rows).unwrap(),
            columns: NonZeroUsize::new(columns).unwrap(),
            ..Self::Settings::DEFAULT
        };
        vec![
            Self::Settings::DEFAULT,
            block(2, 2),
            block(8, 4),
            block(4, 8),
        ]
    }

    /// The block the settings size, reading the rhs along the axis its buffer runs. It asks
    /// nothing of the hardware, so every device runs it.
    fn candidates(
        _hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let rhs = match item.layouts.rhs {
            MatmulLayout::Col => MatmulOrder::InBufferOrder,
            MatmulLayout::Row | MatmulLayout::Strided => MatmulOrder::Stated,
        };
        core::iter::once(Self {
            settings: *settings,
            orders: MatmulOperands {
                rhs,
                ..MatmulOperands::all(MatmulOrder::Stated)
            },
            axes: item.axes,
        })
    }

    /// The lhs's step is the block's depth down `K`, and the rhs's its extent along its own
    /// contiguous axis. A col rhs's step is contracted with the lhs's, a line of each summed into a cell
    /// one value wide, so the output is written a value at a time; a row rhs's lands in the
    /// output's cells, lined with them, and the output is written the block's columns at a time.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        let (rhs_partner, out) = match self.orders.rhs {
            MatmulOrder::InBufferOrder => (MatmulOperand::Lhs, 1),
            MatmulOrder::Stated => (MatmulOperand::Out, self.extent(MatmulDim::N)),
        };
        for (operand, elements) in [
            (MatmulOperand::Lhs, self.extent(MatmulDim::K)),
            (
                MatmulOperand::Rhs,
                self.extent(self.orders.innermost(MatmulOperand::Rhs)),
            ),
            (MatmulOperand::Out, out),
        ] {
            vectorization.require(VectorRequirement::Access { operand, elements });
        }
        vectorization.require(VectorRequirement::Together {
            operand: MatmulOperand::Rhs,
            with: rhs_partner,
        });
    }

    /// It holds its block in registers and contracts it with the device's vector arithmetic: the
    /// work. Its steps' reads are its routine's to count, which places its units.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: self.registers_per_unit(item),
        });
    }
}

impl MatmulMicrokernel for UnitRegistersMicrokernel {
    /// A unit owns a block of the rows.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Tiled;

    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.orders
    }
}
