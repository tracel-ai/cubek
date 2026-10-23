use crate::matmul::dense::plane::PlaneShuffleMicrokernel;
use crate::matmul::{
    MatmulAxes, MatmulDim, MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands,
    MatmulOrder, MatmulRows, MatmulRowsKind, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Instruction, Levels, RegisterBlock};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// Every row of a run of the output's columns, owned by one unit that walks all of `K` alone, so
/// no partial meets another unit's. What a product of few rows runs on a device that commits to
/// no plane width, where the plane shuffle cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWholeKMicrokernel {
    pub settings: UnitWholeKMicrokernelSettings,
    /// Every row of the product, held in registers. An extent in a blueprint, an exception to the
    /// crate's rule, since each row count compiles a kernel of its own.
    pub rows: usize,
    /// The widest vector the device reads the weight in, which sizes the accumulator's register
    /// budget before the bound width is known.
    pub rhs_vector_ceiling: usize,
    /// The order each operand is read in: the weight along `K` for a col weight, along `N` for a
    /// row weight; the others as the product names them.
    pub orders: MatmulOperands<MatmulOrder>,
    /// The caller's axes its levels are stated along.
    pub axes: MatmulAxes,
}

/// One unit's step, `columns × depth`, counted in reads rather than elements, so one setting loads
/// the same bytes a step at every element size and on every device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWholeKMicrokernelSettings {
    /// The output's columns one unit reads a step.
    pub columns: NonZeroUsize,
    /// The steps of `K` one unit reads a step, their loads issued together.
    pub depth: NonZeroUsize,
}

impl UnitWholeKMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        columns: NonZeroUsize::MIN,
        depth: NonZeroUsize::new(8).unwrap(),
    };

    /// The default's columns, `depth` steps of `K` a step.
    pub const fn deep(depth: usize) -> Self {
        Self {
            depth: NonZeroUsize::new(depth).unwrap(),
            ..Self::DEFAULT
        }
    }
}

impl core::fmt::Display for UnitWholeKMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "block={}x{}", self.columns, self.depth)
    }
}

impl UnitWholeKMicrokernel {
    /// Steps of `K` a unit holds at once: it loads a step once the last is summed.
    pub const STAGES_IN_FLIGHT: usize = 1;

    /// The plane shuffle's, so the few-rows table's row bound is one number for both.
    pub const MAX_ROWS: usize = PlaneShuffleMicrokernel::MAX_ROWS;

    /// What a product of no rows, or of more than [`MAX_ROWS`](Self::MAX_ROWS), is declined with.
    pub const ROWS_OUT_OF_RANGE: Refusal = Refusal::Declined {
        why: "the unit whole-K microkernel holds from one row to `UnitWholeKMicrokernel::MAX_ROWS`, every one against one read of the weight",
    };

    /// What a weight lined along neither of its axes is declined with.
    pub const STRIDED_RHS: Refusal = Refusal::Declined {
        why: "the unit whole-K microkernel reads a weight lined along `K` or `N`",
    };

    /// What a col weight bound one element at a time is declined with: the tile DSL reads it in
    /// vectors of at least two.
    pub const COL_RHS_NEEDS_VECTORS_ALONG_K: Refusal = Refusal::Declined {
        why: "the unit whole-K microkernel reads a col weight and the lhs beside it in vectors of at least two elements along `K`",
    };

    /// What a col weight is declined with where a unit holds more than one column: the tile DSL
    /// sums its partials correctly only for one.
    pub const COL_RHS_ONE_COLUMN_A_UNIT: Refusal = Refusal::Declined {
        why: "the unit whole-K microkernel reads a col weight one column a unit, each folding its line of partials",
    };

    /// The order a weight lying as `layout` is read in, none for one lined along neither axis.
    fn order_of(layout: MatmulLayout) -> Option<MatmulOrder> {
        match layout {
            MatmulLayout::Col => Some(MatmulOrder::InBufferOrder),
            MatmulLayout::Row => Some(MatmulOrder::Stated),
            MatmulLayout::Strided => None,
        }
    }

    /// Whether the operands bound at `widths` are read in vectors this microkernel merges.
    ///
    /// # Errors
    ///
    /// [`COL_RHS_NEEDS_VECTORS_ALONG_K`](Self::COL_RHS_NEEDS_VECTORS_ALONG_K), for a col weight
    /// bound one element at a time, and
    /// [`COL_RHS_ONE_COLUMN_A_UNIT`](Self::COL_RHS_ONE_COLUMN_A_UNIT), for a col weight under a
    /// block of more than one column.
    pub fn check_vectors(&self, widths: &MatmulOperands<usize>) -> Result<(), Refusal> {
        match (self.orders.rhs, widths.rhs, self.settings.columns.get()) {
            (MatmulOrder::InBufferOrder, 0 | 1, _) => Err(Self::COL_RHS_NEEDS_VECTORS_ALONG_K),
            (MatmulOrder::InBufferOrder, _, 2..) => Err(Self::COL_RHS_ONE_COLUMN_A_UNIT),
            (MatmulOrder::InBufferOrder | MatmulOrder::Stated, _, _) => Ok(()),
        }
    }

    /// Elements of the output's columns one unit reads a step: its block's columns, each a vector
    /// where the weight runs along `N`.
    pub fn columns(&self) -> usize {
        self.settings.columns.get()
            * match self.orders.rhs {
                MatmulOrder::InBufferOrder => 1,
                MatmulOrder::Stated => self.rhs_vector_ceiling,
            }
    }

    /// Elements of `K` one unit reads a step: its block's depth, each a vector where the weight
    /// runs along `K`.
    pub fn depth(&self) -> usize {
        self.settings.depth.get()
            * match self.orders.rhs {
                MatmulOrder::InBufferOrder => self.rhs_vector_ceiling,
                MatmulOrder::Stated => 1,
            }
    }

    /// The levels inside one unit, which a routine states the rest of the partitioning around:
    /// one step of its block, walked through all of `K`.
    pub fn tiling(&self) -> Levels {
        let MatmulAxes { m, n, k } = self.axes;
        Levels::leaf(&[(m, self.rows), (n, self.columns()), (k, self.depth())]).walk_every(&[k])
    }

    /// The extent along `dim` of one unit's step: the leaf of [`tiling`](Self::tiling).
    pub fn leaf_extent(&self, dim: MatmulDim) -> usize {
        match dim {
            MatmulDim::M => self.rows,
            MatmulDim::N => self.columns(),
            MatmulDim::K => self.depth(),
        }
    }

    /// 32-bit registers one unit keeps: its accumulator, summed in `item.sum`, and one step's
    /// block of the lhs and the weight. The compiler adds addressing and loop state on top.
    pub fn registers_per_unit(&self, item: &MatmulWorkItem) -> u32 {
        let [rows, columns, depth] = [self.rows, self.columns(), self.depth()];
        let bytes = self.accumulator_scalars() * item.sum.size()
            + rows * depth * item.elements.lhs.size()
            + columns * depth * item.elements.rhs.size();
        bytes.div_ceil(4) as u32
    }

    /// Scalars one unit's accumulator holds: a cell per row and column, which for a col weight is
    /// a line of partials summed on drain.
    fn accumulator_scalars(&self) -> usize {
        let cell = match self.orders.rhs {
            MatmulOrder::InBufferOrder => self.rhs_vector_ceiling,
            MatmulOrder::Stated => 1,
        };
        self.rows * self.columns() * cell
    }

    /// The instruction a unit's sum opens on: a register block holding all of its accumulator.
    pub fn tile_instruction(&self) -> Instruction {
        Instruction::Registers {
            config: RegisterBlock::new(self.accumulator_scalars()),
        }
    }
}

/// Its rows, its settings, and how the weight lies.
impl core::fmt::Display for UnitWholeKMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let weight = match self.orders.rhs {
            MatmulOrder::InBufferOrder => "col",
            MatmulOrder::Stated => "row",
        };
        write!(f, "rows={} {} rhs={weight}", self.rows, self.settings)
    }
}

impl Microkernel for UnitWholeKMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = UnitWholeKMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = ();

    const DECLINE: &'static str = "the unit whole-K microkernel asks nothing of the hardware";

    /// The default depth, then a shallower and a deeper one: a deeper step issues more loads
    /// together and holds more registers.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::deep(4),
            Self::Settings::deep(16),
        ]
    }

    /// From one row to [`MAX_ROWS`](Self::MAX_ROWS), every one held whole, and a weight lined
    /// along one of its axes.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.rows {
            MatmulRows::Whole(rows) if (1..=Self::MAX_ROWS).contains(&rows) => {}
            MatmulRows::Whole(_) | MatmulRows::Tiled => return Err(Self::ROWS_OUT_OF_RANGE),
        }
        match item.layouts.rhs {
            MatmulLayout::Col | MatmulLayout::Row => Ok(()),
            MatmulLayout::Strided => Err(Self::STRIDED_RHS),
        }
    }

    /// The one the settings size. Its units share nothing, so it asks nothing of the hardware
    /// but the width it reads the weight in.
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let rows = match item.rows {
            MatmulRows::Whole(rows) if (1..=Self::MAX_ROWS).contains(&rows) => Some(rows),
            MatmulRows::Whole(_) | MatmulRows::Tiled => None,
        };
        rows.zip(Self::order_of(item.layouts.rhs))
            .map(|(rows, rhs)| Self {
                settings: *settings,
                rows,
                rhs_vector_ceiling: hw.vector_ceiling(item.elements.rhs),
                orders: MatmulOperands {
                    rhs,
                    ..MatmulOperands::all(MatmulOrder::Stated)
                },
                axes: item.axes,
            })
            .into_iter()
    }

    /// Each operand's step is the leaf's extent along the axis it is bound innermost along: the lhs
    /// down `K`, the weight along its own contiguous axis, the output along `N`. A col weight's
    /// step is contracted with the lhs's, a line of each at a time; a row weight's lands in the output's
    /// cells, lined with it.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        for operand in MatmulOperand::ALL {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: self.leaf_extent(self.orders.innermost(operand)),
            });
        }
        let rhs_partner = match self.orders.rhs {
            MatmulOrder::InBufferOrder => MatmulOperand::Lhs,
            MatmulOrder::Stated => MatmulOperand::Out,
        };
        vectorization.require(VectorRequirement::Together {
            operand: MatmulOperand::Rhs,
            with: rhs_partner,
        });
    }

    /// It holds its sums in registers and contracts them with the device's vector arithmetic: the
    /// work. The reads are its routine's to count, which places its units.
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

impl MatmulMicrokernel for UnitWholeKMicrokernel {
    /// It holds every row, compiled in, against each step of the weight.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.orders
    }
}
