use super::{ConvProblem, ConvRoles, ConvSettings};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{Axis, Levels, Partitioning, Space, StageStorage};
use cubek_microkernel::matmul::plane::{PlaneFragmentInstruction, PlaneFragmentMicrokernel};
use cubek_microkernel::matmul::{MatmulDim, MatmulOperand, MatmulOperands};
use cubek_std::microkernel::{CubeStaging, GlobalStream, StageRing};

/// The ring every conv stage is held in.
pub const STAGE_RING: StageRing = StageRing::SINGLE;

/// What one step of the contraction, the instruction's `k` cells, is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvStep {
    /// The contracted axis alone: a whole step is `k` cells of it, the last one cut short.
    Alone { k: usize },
    /// A few cells of the contracted axis, rounded up to a power of two, and as many of the
    /// second contracted axis as fill the rest of the step.
    Beside { beside: usize, cells: usize },
}

impl ConvStep {
    pub fn new(problem: &ConvProblem, k: usize) -> Self {
        let cells = problem.extent(problem.operation.roles().axes().k);
        match (k / cells.next_power_of_two()).max(1) {
            1 => Self::Alone { k },
            beside => Self::Beside {
                beside,
                cells: k / beside,
            },
        }
    }

    fn beside(self) -> usize {
        match self {
            Self::Alone { .. } => 1,
            Self::Beside { beside, .. } => beside,
        }
    }

    fn cells(self) -> usize {
        match self {
            Self::Alone { k } => k,
            Self::Beside { cells, .. } => cells,
        }
    }

    /// The axis a stage walks and the steps along it: the contracted axis alone, or the second
    /// one where it shares the step.
    fn walk(self, roles: &ConvRoles, problem: &ConvProblem) -> (Axis, usize) {
        match self {
            Self::Alone { k } => {
                let axis = roles.axes().k;
                (axis, problem.extent(axis).div_ceil(k))
            }
            Self::Beside { beside, .. } => {
                let axis = roles.beside();
                (axis, problem.extent(axis).div_ceil(beside))
            }
        }
    }
}

/// The cuts a routine states over the output's rows and columns, and over the contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTiles {
    roles: ConvRoles,
    /// One instruction's rows and columns, and its contraction depth.
    instruction: (usize, usize, usize),
    /// Instructions a plane holds along the rows and the columns.
    fragments: (usize, usize),
    /// Planes along the same two axes.
    planes: (usize, usize),
    step: ConvStep,
    stage_steps: usize,
    /// Elements one read of a stage fill brings along each input's innermost axis.
    runs: MatmulOperands<usize>,
}

impl ConvTiles {
    /// The tiles `settings` ask of `problem`, over the instruction `plane` contracts through.
    ///
    /// # Errors
    ///
    /// [`CMMA_STEP_ACROSS_TAPS`], for a step of taps and channels under cmma.
    pub fn new(
        problem: &ConvProblem,
        plane: &PlaneFragmentMicrokernel,
        settings: &ConvSettings,
    ) -> Result<Self, Refusal> {
        let [(_, m), (_, n), (_, k)] = plane.instruction.shape();
        let step = ConvStep::new(problem, k);
        let cmma = matches!(plane.instruction, PlaneFragmentInstruction::Cmma { .. });
        if cmma
            && step.beside() > 1
            && let Some(refusal) = problem.operation.cmma_step_declined()
        {
            return Err(refusal);
        }
        let mut tiles = Self {
            roles: problem.operation.roles(),
            instruction: (m, n, k),
            fragments: (plane.settings.along_m.get(), plane.settings.along_n.get()),
            planes: settings.planes,
            step,
            stage_steps: settings.stage_steps,
            runs: MatmulOperands::all(1),
        };
        let cells = |operand| settings.lines.cells(tiles.run(operand));
        tiles.runs = MatmulOperands {
            lhs: cells(MatmulOperand::Lhs),
            rhs: cells(MatmulOperand::Rhs),
            out: 1,
        };
        Ok(tiles)
    }

    /// The cells of `operand` one instruction tile holds along its innermost axis.
    fn run(&self, operand: MatmulOperand) -> usize {
        let (m, n, _) = self.instruction;
        match self.roles.axes().dim(self.roles.inner.get(operand)) {
            Some(MatmulDim::M) => m,
            Some(MatmulDim::N) => n,
            Some(MatmulDim::K) => self.step.cells(),
            None => 1,
        }
    }

    fn leaf(&self, stage: (Axis, usize)) -> Levels {
        let (m, n, _) = self.instruction;
        let axes = self.roles.axes();
        Levels::leaf(&[
            (axes.m, m),
            (axes.n, n),
            (self.roles.beside(), self.step.beside()),
            (axes.k, self.step.cells()),
        ])
        .walk(&[(axes.m, self.fragments.0), (axes.n, self.fragments.1)])
        .walk(&[stage])
    }
}

/// How a launch places its contraction over the problem: a cube for every box of rows and
/// columns, its planes each owning a tile of it, the box walking every contracted cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvPartitioning {
    partitioning: Partitioning,
    runs: MatmulOperands<usize>,
    /// Rows and columns of the output one stage owner holds, and the contraction cells a stage
    /// holds, `(rows, columns, depth)`.
    stage: (usize, usize, usize),
    /// Elements one write brings along the output's innermost axis.
    out_run: usize,
    /// Planes in a cube.
    planes: usize,
    /// Contraction cells one owner walks over its box, the cut steps counted whole.
    contracted: usize,
}

impl ConvPartitioning {
    /// A cube stages for all its planes.
    pub fn cube(problem: &ConvProblem, tiles: &ConvTiles) -> Self {
        let (planes_m, planes_n) = tiles.planes;
        let (m, n, _) = tiles.instruction;
        let roles = tiles.roles;
        let axes = roles.axes();
        let (stage, steps) = Self::stage(problem, tiles);
        let levels = tiles
            .leaf(stage)
            .planes(&[(axes.m, planes_m), (axes.n, planes_n)])
            .walk_every(&roles.contracted)
            .cubes(&[axes.n, axes.m, roles.across()])
            .batches(&[roles.batch()])
            .build();
        let rows = m * tiles.fragments.0 * planes_m;
        let columns = n * tiles.fragments.1 * planes_n;
        let contracted: usize = roles
            .contracted
            .iter()
            .map(|&axis| {
                let extent = problem.extent(axis);
                match (axis == roles.beside(), axis == axes.k) {
                    (true, _) => extent.next_multiple_of(tiles.step.beside()),
                    (_, true) => extent.next_multiple_of(tiles.step.cells()),
                    _ => extent,
                }
            })
            .product();
        Self {
            partitioning: Partitioning::new(problem.space(), levels),
            runs: tiles.runs,
            out_run: match axes.dim(roles.inner.out) {
                Some(MatmulDim::M) => rows,
                Some(MatmulDim::N) => columns,
                _ => 1,
            },
            stage: (rows, columns, steps),
            planes: tiles.planes.0 * tiles.planes.1,
            contracted,
        }
    }

    /// The axis a stage walks and the steps along it, and the contraction cells that holds.
    fn stage(problem: &ConvProblem, tiles: &ConvTiles) -> ((Axis, usize), usize) {
        let (_, _, k) = tiles.instruction;
        let (axis, total) = tiles.step.walk(&tiles.roles, problem);
        // A stage never overhangs: the most steps up to the setting's that divide the walk.
        let steps = (1..=tiles.stage_steps.min(total))
            .rev()
            .find(|steps| total % steps == 0)
            .unwrap_or(1);
        ((axis, steps), steps * k)
    }

    pub fn as_tile(&self) -> &Partitioning {
        &self.partitioning
    }

    /// The problem's space.
    pub fn space(&self) -> &Space {
        self.partitioning.space()
    }

    /// Whether the tiles overhang `axis`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, axis: Axis) -> bool {
        self.partitioning.overhangs(axis)
    }

    /// Planes in a cube.
    pub fn planes(&self) -> usize {
        self.planes
    }

    /// How the stages hold their cells: tiles of the innermost box, in order.
    pub fn storage(&self) -> StageStorage {
        StageStorage::Tiled {
            block: self.partitioning.leaf().extents(),
            chunks: RowChunks::InOrder,
        }
    }

    /// Bytes of shared memory a cube's stages hold, the operands in `elements`.
    pub fn stage_bytes(&self, elements: &MatmulOperands<ElemType>) -> usize {
        let (rows, columns, depth) = self.stage;
        let cells = rows * elements.lhs.size() + columns * elements.rhs.size();
        STAGE_RING.stages_in_flight.get() * depth * cells
    }

    /// What one cube moves and computes: the cube's box of the output read over the whole
    /// contraction and written once, the cells its edge cuts short counted as spent.
    pub fn staging(&self, elements: &MatmulOperands<ElemType>) -> CubeStaging {
        let (rows, columns, depth) = self.stage;
        let (rows, columns) = (rows as u64, columns as u64);
        let contracted = self.contracted as u64;
        let size = |operand| elements.get(operand).size();
        let stream = |bytes: u64, run: usize| GlobalStream {
            bytes,
            run_bytes: run,
        };
        CubeStaging {
            operations: (2 * rows * columns * contracted) as f64,
            reads: vec![
                stream(
                    rows * contracted * size(MatmulOperand::Lhs) as u64,
                    self.runs.lhs * size(MatmulOperand::Lhs),
                ),
                stream(
                    columns * contracted * size(MatmulOperand::Rhs) as u64,
                    self.runs.rhs * size(MatmulOperand::Rhs),
                ),
            ],
            writes: vec![stream(
                rows * columns * size(MatmulOperand::Out) as u64,
                self.out_run * size(MatmulOperand::Out),
            )],
            stages: self.contracted.div_ceil(depth) as u64,
            stages_in_flight: STAGE_RING.stages_in_flight.get() as u32,
            drain_atomics: 0,
        }
    }

    /// What a stage fill's own reads require of the inputs' widths: runs along each one's
    /// innermost axis.
    pub fn vectorize(&self, vectorization: &mut Vectorization<MatmulOperand>) {
        for operand in [MatmulOperand::Lhs, MatmulOperand::Rhs] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: self.runs.get(operand),
            });
        }
    }
}
