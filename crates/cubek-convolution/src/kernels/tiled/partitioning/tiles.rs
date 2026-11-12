use super::{ConvBlock, ConvStep};
use crate::kernels::tiled::{ConvProblem, ConvRoles, ConvSettings};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{Axis, Level, Levels, Partitioning, StageStorage};
use cubek_microkernel::matmul::plane::{PlaneFragmentInstruction, PlaneFragmentMicrokernel};
use cubek_microkernel::matmul::{MatmulDim, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;

/// The rows, columns and contraction depth of one matrix instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionShape {
    pub rows: usize,
    pub columns: usize,
    pub depth: usize,
}

impl From<&PlaneFragmentInstruction> for InstructionShape {
    fn from(instruction: &PlaneFragmentInstruction) -> Self {
        let [(_, rows), (_, columns), (_, depth)] = instruction.shape();
        Self {
            rows,
            columns,
            depth,
        }
    }
}

/// The cuts a routine states over the output's rows and columns, and over the contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTiles {
    roles: ConvRoles,
    instruction: InstructionShape,
    settings: ConvSettings,
    orders: MatmulOperands<BindingOrder>,
    step: ConvStep,
    /// Elements one read of a stage fill brings along each input's innermost axis.
    runs: MatmulOperands<usize>,
}

impl ConvTiles {
    /// The tiles `settings` ask of `problem`, over the instruction `plane` contracts through.
    ///
    /// # Errors
    ///
    /// [`CMMA_STEP_ACROSS_TAPS`](crate::kernels::tiled::CMMA_STEP_ACROSS_TAPS), for a step of taps
    /// and channels under cmma.
    pub fn new(
        problem: &ConvProblem,
        plane: &PlaneFragmentMicrokernel,
        settings: &ConvSettings,
    ) -> Result<Self, Refusal> {
        let instruction = InstructionShape::from(&plane.instruction);
        let step = ConvStep::new(problem, instruction.depth);
        let cmma = matches!(plane.instruction, PlaneFragmentInstruction::Cmma { .. });
        if cmma && let Some(refusal) = step.cmma_declined(&problem.roles()) {
            return Err(refusal);
        }
        let mut tiles = Self {
            roles: problem.roles(),
            instruction,
            settings: *settings,
            orders: plane.orders,
            step,
            runs: MatmulOperands::all(1),
        };
        let cells = |operand| settings.vectors.cells(tiles.run(operand));
        tiles.runs = MatmulOperands {
            lhs: cells(MatmulOperand::Lhs),
            rhs: cells(MatmulOperand::Rhs),
            out: tiles.run(MatmulOperand::Out),
        };
        Ok(tiles)
    }

    /// The cells of `operand` one instruction tile holds along its innermost axis.
    fn run(&self, operand: MatmulOperand) -> usize {
        match self.roles.axes().dim(self.roles.inner.get(operand)) {
            Some(MatmulDim::M) => self.instruction.rows,
            Some(MatmulDim::N) => self.instruction.columns,
            Some(MatmulDim::K) => self.step.cells(),
            None => 1,
        }
    }

    /// What one cube covers of the output, with a stage `depth` cells deep.
    pub(super) fn block(&self, depth: usize) -> ConvBlock {
        let (fragments, planes) = (
            self.settings.microkernel.plane,
            self.settings.microkernel.planes,
        );
        ConvBlock {
            rows: self.instruction.rows * fragments.along_m.get() * planes.along_m.get(),
            across: self.settings.across.get(),
            columns: self.instruction.columns * fragments.along_n.get() * planes.along_n.get(),
            depth,
        }
    }

    /// The edge of the innermost box along each axis.
    fn leaf_block(&self) -> [(Axis, usize); 4] {
        let axes = self.roles.axes();
        [
            (axes.m, self.instruction.rows),
            (axes.n, self.instruction.columns),
            (self.roles.contracted.beside, self.step.beside()),
            (axes.k, self.step.cells()),
        ]
    }

    /// How the stages hold their cells: tiles of the innermost box `problem`'s launch cuts, in
    /// order. Built once, so the stage is sized with the very storage that is launched.
    pub(super) fn storage(&self, problem: &ConvProblem) -> StageStorage {
        let (axis, _) = self.step.walk(&self.roles, problem);
        let partitioning = Partitioning::new(problem.space(), self.levels((axis, 1)));
        StageStorage::Tiled {
            block: partitioning.leaf().extents(),
            chunks: RowChunks::InOrder,
        }
    }

    /// Bytes of shared memory a single stage of `steps` instruction steps holds on `hw` laid out as
    /// `storage`.
    pub(super) fn stage_bytes(
        &self,
        hw: &Hardware,
        elements: MatmulOperands<ElemType>,
        storage: &StageStorage,
        steps: usize,
    ) -> usize {
        self.block(steps * self.instruction.depth)
            .stage(self.roles.axes())
            .bytes(hw, elements, storage, self.orders)
    }

    /// The axis a stage walks and the steps along it, and the contraction cells that holds: the
    /// most steps up to the setting's that divide the walk (a stage never overhangs) and fit
    /// `budget` of shared memory laid out as `storage`.
    pub(super) fn stage(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
        storage: &StageStorage,
        budget: usize,
    ) -> ((Axis, usize), usize) {
        let (axis, total) = self.step.walk(&self.roles, problem);
        // One stage holds a single ring, and the bounce window shares those bytes (the larger of
        // the two is held), so neither counts here.
        let fits = |steps: usize| self.stage_bytes(hw, problem.elements, storage, steps) <= budget;
        let divisor = (1..=self.settings.stage_steps.get().min(total))
            .rev()
            .find(|&steps| total % steps == 0 && fits(steps))
            .unwrap_or(1);
        // A walk with no divisor in reach (a prime one) would run a stage a step, so its whole
        // walk is one stage where that fits.
        let whole = (divisor == 1 && total > 1)
            .then_some(total)
            .filter(|&steps| fits(steps));
        let steps = whole.unwrap_or(divisor);
        ((axis, steps), steps * self.instruction.depth)
    }

    /// The contraction cells one owner walks over its box of `problem`, the cut steps counted
    /// whole.
    pub(super) fn contracted(&self, problem: &ConvProblem) -> usize {
        let (beside, k) = (self.roles.contracted.beside, self.roles.axes().k);
        self.roles
            .contracted
            .in_order()
            .iter()
            .map(|&axis| {
                let extent = problem.extent(axis);
                match (axis == beside, axis == k) {
                    (true, _) => extent.next_multiple_of(self.step.beside()),
                    (_, true) => extent.next_multiple_of(self.step.cells()),
                    _ => extent,
                }
            })
            .product()
    }

    /// Elements one read or write of a launch of `block` brings along each operand's innermost
    /// axis: the stage fill's runs, and the output's, which a cube writes by its whole box.
    pub(super) fn runs(&self, block: &ConvBlock) -> MatmulOperands<usize> {
        let out = match self.roles.axes().dim(self.roles.inner.out) {
            Some(MatmulDim::M) => block.rows,
            Some(MatmulDim::N) => block.columns,
            Some(MatmulDim::K) | None => 1,
        };
        MatmulOperands { out, ..self.runs }
    }

    /// The levels of a launch whose stages walk `stage`: its planes' tiles, every contracted cell
    /// walked, a cube for every box of the output and one for every batch.
    pub(super) fn levels(&self, stage: (Axis, usize)) -> Vec<Level> {
        let axes = self.roles.axes();
        let planes = self.settings.microkernel.planes;
        self.leaf(stage)
            .planes(&[
                (axes.m, planes.along_m.get()),
                (axes.n, planes.along_n.get()),
                (self.roles.free.across, self.settings.across.get()),
            ])
            .walk_every(&self.roles.contracted.in_order())
            .cubes(&[axes.n, axes.m, self.roles.free.across])
            .batches(&[self.roles.free.batch])
            .build()
    }

    fn leaf(&self, stage: (Axis, usize)) -> Levels {
        let axes = self.roles.axes();
        let fragments = self.settings.microkernel.plane;
        Levels::leaf(&self.leaf_block())
            .walk(&[
                (axes.m, fragments.along_m.get()),
                (axes.n, fragments.along_n.get()),
            ])
            .walk(&[stage])
    }
}
