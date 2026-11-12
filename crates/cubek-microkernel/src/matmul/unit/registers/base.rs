use crate::matmul::plane::PlaneShuffleMicrokernel;
use crate::matmul::{
    MatmulAxes, MatmulDim, MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands,
    MatmulRows, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::{Levels, SoftwareMatmul};
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};
use cubek_std::{Hardware, UNIT_REGISTERS};

/// A block of the output one unit owns, summed in registers while it walks `K`, as its settings
/// size it. It asks nothing of the hardware, so it is its settings and the order it reads the rhs
/// in. Where its work item holds every row whole, the block holds all of them, compiled in.
///
/// It reads the rhs along the axis its buffer runs: a col rhs down `K`, a line of it merged with a
/// line of the lhs into each cell, and a row rhs, or one lined along neither axis, along `N`,
/// lined with the output's cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegistersMicrokernel {
    pub settings: UnitRegistersMicrokernelSettings,
    /// The block's rows: the settings', or every row of a work item that holds them whole.
    pub rows: usize,
    /// The order each operand is read in.
    pub orders: MatmulOperands<BindingOrder>,
    /// The caller's axes its levels are stated along.
    pub axes: MatmulAxes,
}

/// The size of the block a unit owns, and how deep one step of its walk reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegistersMicrokernelSettings {
    /// The block's rows, where its work item tiles them.
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
    /// The most rows a block holds where it holds every row whole: the plane shuffle's, so the
    /// few-rows autotune's row bound is one number for both.
    const MAX_WHOLE_ROWS: usize = PlaneShuffleMicrokernel::MAX_ROWS;

    /// What a work item holding no rows, or more than a block holds whole, is declined with.
    pub const WHOLE_ROWS_OUT_OF_RANGE: Refusal = Refusal::Declined {
        why: "the unit microkernel holds from one to `PlaneShuffleMicrokernel::MAX_ROWS` rows whole, every one against one read of the rhs",
    };

    /// 32-bit registers one unit keeps its block in: its accumulator, and one step's rows of the
    /// lhs and columns of the rhs. A lower bound: the compiler adds addressing and loop state.
    pub fn registers_per_unit(&self, item: &MatmulWorkItem) -> u32 {
        let [rows, columns, depth] = self.extents();
        let bytes = rows * columns * item.sum.size()
            + rows * depth * item.elements.lhs.size()
            + depth * columns * item.elements.rhs.size();
        bytes.div_ceil(4) as u32
    }

    /// The register block a unit's sum opens as: its accumulator's cells, unrolled while they fit
    /// a unit's registers. A block past them, such as an attention unit's sum over a 512-wide
    /// value dim, walks its cells at runtime.
    pub fn block(&self) -> SoftwareMatmul {
        SoftwareMatmul::new(UNIT_REGISTERS as usize)
    }

    /// [`block`](Self::block) with partials: with a col rhs, each cell keeps one partial sum per
    /// lane of a `K` line and reduces them once at the drain, instead of a horizontal sum every
    /// step. Only for an accumulator read solely at its drain, like a matmul's output; an
    /// attention score is read by its softmax first and uses `block`.
    pub fn drained_block(&self) -> SoftwareMatmul {
        SoftwareMatmul::new(UNIT_REGISTERS as usize).partials()
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
        [
            self.rows,
            self.settings.columns.get(),
            self.settings.depth.get(),
        ]
    }
}

/// Its block, and the order it reads the rhs in.
impl core::fmt::Display for UnitRegistersMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [rows, columns, depth] = self.extents();
        write!(f, "block={rows}x{columns}x{depth} rhs={}", self.orders.rhs)
    }
}

impl Microkernel for UnitRegistersMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = UnitRegistersMicrokernelSettings;
    type Operand = MatmulOperand;
    type CubeSpace = ();

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

    /// A work item holding every row whole holds from one to
    /// `MAX_WHOLE_ROWS` of them; one that tiles them, any.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.rows {
            MatmulRows::Whole(rows) if !(1..=Self::MAX_WHOLE_ROWS).contains(&rows) => {
                Err(Self::WHOLE_ROWS_OUT_OF_RANGE)
            }
            MatmulRows::Whole(_) | MatmulRows::Tiled => Ok(()),
        }
    }

    /// The block the settings size, every row of a work item that holds them whole, reading the
    /// rhs along the axis its buffer runs. It asks nothing of the hardware, so every device runs
    /// it.
    fn candidates(
        _hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let rhs = match item.layouts.rhs {
            MatmulLayout::Col => BindingOrder::InBufferOrder,
            MatmulLayout::Row | MatmulLayout::Strided => BindingOrder::Stated,
        };
        let rows = match item.rows {
            MatmulRows::Whole(rows) => rows,
            MatmulRows::Tiled => settings.rows.get(),
        };
        core::iter::once(Self {
            settings: *settings,
            rows,
            orders: MatmulOperands {
                rhs,
                ..MatmulOperands::all(BindingOrder::Stated)
            },
            axes: item.axes,
        })
    }

    /// The lhs's step is the block's depth down `K`, and the rhs's its extent along its own
    /// contiguous axis. A col rhs's step is contracted with the lhs's, a line of each summed into a
    /// cell one value wide, so the output is written a value at a time; a row rhs's lands in the
    /// output's cells, lined with them, and the output is written the block's columns at a time.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        let (rhs_partner, out) = match self.orders.rhs {
            BindingOrder::InBufferOrder => (MatmulOperand::Lhs, 1),
            BindingOrder::Stated => (MatmulOperand::Out, self.extent(MatmulDim::N)),
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
    /// The levels inside one unit's block, which a routine states the rest of the partitioning
    /// around: one step of the block, walked through all of `K`.
    fn tiling(&self) -> Levels {
        let [rows, columns, depth] = self.extents();
        let MatmulAxes { m, n, k } = self.axes;
        Levels::leaf(&[(m, rows), (n, columns), (k, depth)]).walk_every(&[k])
    }

    fn orders(&self) -> MatmulOperands<BindingOrder> {
        self.orders
    }
}
