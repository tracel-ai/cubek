use super::{ConvBlock, ConvTiles};
use crate::kernels::tiled::{ConvProblem, ConvRoles};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::space::PlaneOrder;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{Dimension, Partitioning, Space, StageOrder, StageStorage};
use cubek_microkernel::matmul::unit::UnitRegistersMicrokernel;
use cubek_microkernel::matmul::{MatmulDim, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream, StageRing};

/// How a launch's boxes lie over its problem, along the rows, the positions across and the
/// columns: what a caller measuring the settings judges a setting's waste by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvCoverage {
    /// Cells the boxes cover, a box overhanging an edge counted whole.
    pub covered_cells: usize,
    /// Cells of the problem the boxes cover.
    pub problem_cells: usize,
    /// Cubes the grid spawns: one a box, for every outer position.
    pub cube_count: usize,
}

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
                chunks: RowChunks::InOrder,
                order: StageOrder::AsSource,
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
        let dimensions = roles.dimensions();
        let levels = microkernel
            .block_tiling()
            .units_distributed(dimensions.n, blocks_per_plane)
            .launch_planes(&roles.plane_dimensions(), order)
            .build();
        let (rows, columns) = (microkernel.rows, microkernel.settings.columns.get());
        let depth = microkernel.settings.depth.get();
        let contracted = roles
            .contracted
            .in_order()
            .iter()
            .map(|&dimension| problem.extent(dimension))
            .product();
        let along = |dimension: Dimension| match dimensions.dim(dimension) {
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
            storage: StageStorage::STRIDED,
            contracted,
        }
    }

    /// The partitioning the tile engine places the kernel's cubes and planes by.
    pub fn tile_partitioning(&self) -> &Partitioning {
        &self.partitioning
    }

    /// The partitioning a launch compiles: this one with the dimension `roles` leave to the launch
    /// freed, so one kernel serves every extent of it. What the blueprint decides on (overhangs,
    /// widths) is read off the problem's own space, never off this one.
    pub fn kernel(&self, roles: &ConvRoles) -> Partitioning {
        let dynamic: Vec<Dimension> = roles.dynamic().into_iter().collect();
        Partitioning::new(
            self.partitioning.space().clone().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// The problem's space.
    pub fn space(&self) -> &Space {
        self.partitioning.space()
    }

    /// Whether the tiles overhang `dimension`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, dimension: Dimension) -> bool {
        self.partitioning.overhangs(dimension)
    }

    /// Whether the tiles overhang a dimension the window gathers the input over, where a read past
    /// the real extents lands off the input whatever the padding.
    pub fn overhangs_gathered(&self, roles: &ConvRoles) -> bool {
        roles
            .spatial()
            .flatten()
            .any(|dimension| self.overhangs(dimension))
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
        let boxes = self.boxes();
        let stream = |operand: MatmulOperand| {
            let size = elements.get(operand).size();
            GlobalStream {
                bytes: boxes.get(operand) * size as u64,
                run_bytes: self.runs.get(operand) * size,
            }
        };
        CubeStaging {
            shared_reads: 0,
            shared_writes: 0,
            operations: (2 * boxes.lhs * self.block.columns as u64) as f64,
            reads: vec![stream(MatmulOperand::Lhs), stream(MatmulOperand::Rhs)],
            writes: vec![stream(MatmulOperand::Out)],
            stages: self.stages() as u64,
            stages_in_flight: ring.stages_in_flight.get() as u32,
            drain_atomics: 0,
        }
    }

    /// Values of each operand one cube walks, each read once: its box's positions of the input and
    /// columns of the rhs over the whole contraction, and its box of the output.
    pub fn boxes(&self) -> MatmulOperands<u64> {
        let (rows, columns) = (self.block.positions() as u64, self.block.columns as u64);
        let contracted = self.contracted as u64;
        MatmulOperands {
            lhs: rows * contracted,
            rhs: columns * contracted,
            out: rows * columns,
        }
    }

    /// How this partitioning's boxes lie over `problem`, along the rows, the positions across and
    /// the columns.
    pub fn coverage(&self, problem: &ConvProblem) -> ConvCoverage {
        let roles = problem.roles();
        let dimensions = roles.dimensions();
        let block = self.block;
        let extents = [
            problem.extent(dimensions.m),
            problem.extent(roles.free.across),
            problem.extent(dimensions.n),
        ];
        let cuts = [block.rows, block.across, block.columns];
        let boxes = [0, 1, 2].map(|dimension| extents[dimension].div_ceil(cuts[dimension]));
        ConvCoverage {
            covered_cells: (0..3)
                .map(|dimension| boxes[dimension] * cuts[dimension])
                .product(),
            problem_cells: extents.iter().product(),
            cube_count: boxes.iter().product::<usize>() * problem.outer_extent(),
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
