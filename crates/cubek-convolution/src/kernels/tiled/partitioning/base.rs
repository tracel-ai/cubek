use super::{ConvBlock, ConvTiles};
use crate::kernels::tiled::{ConvProblem, ConvRoles, OH, OW, RH, RW};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::space::PlaneOrder;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{Axis, Partitioning, Space, StageStorage};
use cubek_microkernel::matmul::unit::UnitRegistersMicrokernel;
use cubek_microkernel::matmul::{MatmulDim, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::autotune::CandidatePriority;
use cubek_std::microkernel::{CubeStaging, GlobalStream, StageRing};

/// How a launch places its contraction over the problem: a cube for every box of rows and
/// columns, its planes each owning a tile of it, the box walking every contracted cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvPartitioning {
    partitioning: Partitioning,
    /// Elements one read or write brings along each operand's innermost axis.
    runs: MatmulOperands<usize>,
    /// What one cube covers, and the contraction cells its stage holds.
    block: ConvBlock,
    /// How the stages hold their cells.
    storage: StageStorage,
    /// Contraction cells one owner walks over its box, the cut steps counted whole.
    contracted: usize,
}

impl ConvPartitioning {
    /// A cube stages for all its planes, its stages within the shared memory `hw` gives a cube.
    pub fn cube_staged(hw: &Hardware, problem: &ConvProblem, tiles: &ConvTiles) -> Self {
        let storage = tiles.storage(problem);
        let (stage, depth) = tiles.stage(hw, problem, &storage, hw.shared_memory_budget());
        let block = tiles.block(depth);
        let partitioning = Partitioning::new(problem.space(), tiles.levels(stage));
        debug_assert_eq!(
            StageStorage::Tiled {
                block: partitioning.leaf().extents(),
                chunks: RowChunks::InOrder
            },
            storage,
            "the stage is sized with the storage the launch holds"
        );
        Self {
            partitioning,
            runs: tiles.runs(&block),
            block,
            storage,
            contracted: tiles.contracted(problem),
        }
    }

    /// A plane of the launch for every box of the output, its units taking `blocks_per_plane` of
    /// them along the columns in turns, each contracting every contracted cell of its block from
    /// global memory in one leaf. Nothing is staged: the box a unit owns is what it reads for.
    pub fn unit_registers(
        problem: &ConvProblem,
        microkernel: &UnitRegistersMicrokernel,
        blocks_per_plane: usize,
        order: PlaneOrder,
    ) -> Self {
        let roles = problem.roles();
        let axes = roles.axes();
        let levels = microkernel
            .block_tiling()
            .units_distributed(axes.n, blocks_per_plane)
            .launch_planes(&roles.plane_axes(), order)
            .build();
        let (rows, columns) = (microkernel.rows, microkernel.settings.columns.get());
        let depth = microkernel.settings.depth.get();
        let contracted = roles
            .contracted
            .in_order()
            .iter()
            .map(|&axis| problem.extent(axis))
            .product();
        let along = |axis: Axis| match axes.dim(axis) {
            Some(MatmulDim::K) => depth,
            Some(MatmulDim::M) => rows,
            Some(MatmulDim::N) => columns,
            None => 1,
        };
        Self {
            partitioning: Partitioning::new(problem.space(), levels),
            runs: MatmulOperands {
                lhs: along(roles.inner.lhs),
                rhs: along(roles.inner.rhs),
                out: along(roles.inner.out),
            },
            block: ConvBlock {
                rows,
                across: 1,
                columns,
                depth,
            },
            storage: StageStorage::Strided,
            contracted,
        }
    }

    pub fn tile_partitioning(&self) -> &Partitioning {
        &self.partitioning
    }

    /// The partitioning a launch compiles: this one with the axis `roles` leave to the launch
    /// freed, so one kernel serves every extent of it. What the blueprint decides on (overhangs,
    /// widths) is read off the problem's own space, never off this one.
    pub fn kernel(&self, roles: &ConvRoles) -> Partitioning {
        let dynamic: Vec<Axis> = roles.dynamic().into_iter().collect();
        Partitioning::new(
            self.partitioning.space().clone().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// The problem's space.
    pub fn space(&self) -> &Space {
        self.partitioning.space()
    }

    /// Whether the tiles overhang `axis`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, axis: Axis) -> bool {
        self.partitioning.overhangs(axis)
    }

    /// Whether the tiles overhang an axis the window gathers the input over, where a read past
    /// the real extents lands off the input whatever the padding.
    pub fn overhangs_gathered(&self) -> bool {
        [OH, OW, RH, RW]
            .into_iter()
            .any(|axis| self.overhangs(axis))
    }

    /// How the stages hold their cells.
    pub fn storage(&self) -> StageStorage {
        self.storage.clone()
    }

    /// Stages one cube walks over the whole contraction.
    pub fn stages(&self) -> usize {
        self.contracted.div_ceil(self.block.depth)
    }

    /// `wanted` where the walk has a second stage to fill ahead, and one stage where it has not.
    pub fn ring(&self, wanted: StageRing) -> StageRing {
        match self.stages() {
            0 | 1 => StageRing::SINGLE,
            _ => wanted,
        }
    }

    /// What one cube covers, and the contraction cells its stage holds.
    pub fn block(&self) -> ConvBlock {
        self.block
    }

    /// What one cube moves and computes: the cube's box of the output read over the whole
    /// contraction and written once, the cells its edge cuts short counted as spent, its stages
    /// held in `ring`.
    pub fn staging(&self, elements: &MatmulOperands<ElemType>, ring: StageRing) -> CubeStaging {
        let (rows, columns) = (self.block.positions() as u64, self.block.columns as u64);
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
                self.runs.out * size(MatmulOperand::Out),
            )],
            stages: self.stages() as u64,
            stages_in_flight: ring.stages_in_flight.get() as u32,
            drain_atomics: 0,
        }
    }

    /// Where a candidate launching this partitioning in cubes of `units_per_cube` units is tried
    /// for `problem` on `hw`: the top band unless its boxes cover at least twice the output or its
    /// grid fills less than half the device, then one band lower. A band is raced whole and a
    /// lower one only when it measures nothing, so the priority only separates the clearly
    /// wasteful from the rest.
    pub fn priority(
        &self,
        problem: &ConvProblem,
        hw: &Hardware,
        units_per_cube: usize,
    ) -> CandidatePriority {
        let roles = problem.roles();
        let axes = roles.axes();
        let block = self.block;
        let extents = [
            problem.extent(axes.m),
            problem.extent(roles.free.across),
            problem.extent(axes.n),
        ];
        let cuts = [block.rows, block.across, block.columns];
        let boxes = [0, 1, 2].map(|axis| extents[axis].div_ceil(cuts[axis]));
        let covered: usize = (0..3).map(|axis| boxes[axis] * cuts[axis]).product();
        let work: usize = extents.iter().product();
        let wasteful = covered >= 2 * work.max(1);
        let cubes = boxes.iter().product::<usize>() * problem.extent(roles.free.batch);
        let idle = hw
            .cubes_to_fill(units_per_cube as u32)
            .is_some_and(|wanted| wanted as usize > 2 * cubes.max(1));
        match wasteful || idle {
            true => CandidatePriority::FIRST.lowered(1),
            false => CandidatePriority::FIRST,
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
