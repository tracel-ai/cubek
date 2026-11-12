use super::block::{BlockDepth, BlockForm};
use super::kernel::unit_registers_conv_kernel;
use super::settings::{UnitBlock, UnitRegistersConvSettings};
use crate::kernels::tiled::{
    ConvBinding, ConvPartitioning, ConvProblem, ConvRoles, ConvRoutine, ConvStaging,
    TileConvBlueprint,
};
use core::num::NonZeroUsize;
use cubecl::prelude::SequenceArg;
use cubecl::tile::Axis;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::ops::matmul::LhsVectorAxis;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_matmul::kernels::{UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings};
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
use cubek_microkernel::matmul::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use cubek_microkernel::matmul::{MatmulDim, MatmulOperand};
use cubek_std::microkernel::{KernelEstimate, Microkernel, StageRing};
use cubek_std::routine::{CubeResources, Grid, Routine};
use cubek_std::{Hardware, UnitDispatch};

/// A count along the rows and a count along the across axis: of planes a cube holds, or of tiles
/// they cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AcrossCounts {
    pub rows: usize,
    pub across: usize,
}

impl AcrossCounts {
    /// The tiles, or the planes, in all.
    pub fn count(self) -> usize {
        self.rows * self.across
    }

    /// The cubes of these planes that cover `tiles`, a tile to each plane of a cube.
    pub fn cubes_over(self, tiles: AcrossCounts) -> usize {
        tiles.rows.div_ceil(self.rows) * tiles.across.div_ceil(self.across)
    }

    /// The planes of a cube, of at most `planes`, that cover these tiles in the fewest cubes, for
    /// cubes spread over the device's workers. Of those it takes the fewest planes, then the most
    /// along the rows, whose tiles are contiguous.
    pub fn spread_split(self, planes: usize) -> AcrossCounts {
        (1..=planes)
            .flat_map(|rows| (1..=planes / rows).map(move |across| AcrossCounts { rows, across }))
            .min_by_key(|split| {
                (
                    split.cubes_over(self),
                    split.count(),
                    std::cmp::Reverse(split.rows),
                )
            })
            .unwrap_or(AcrossCounts { rows: 1, across: 1 })
    }
}

/// How the planes of a cube take the launch's tiles, which the way the device runs a launch's
/// units decides ([`Hardware::unit_dispatch`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaneDeal {
    /// A box of planes, rows by across, each cube one box of tiles, for cubes spread over the
    /// device's workers.
    Boxed(AcrossCounts),
    /// `units` one-unit planes a cube, the launch's planes taking the tiles in row order. A unit
    /// sweeping every cube then takes every `units`-th tile, so no unit sweeps more than one tile
    /// past another, and a run of edge tiles along that order falls to the units in turn.
    InTurns { units: usize },
}

impl PlaneDeal {
    /// The deal `dispatch` runs over `tiles` along the rows and the across axis and `launch_tiles`
    /// in all, in cubes of at most the planes of `planes`.
    pub fn of(
        dispatch: UnitDispatch,
        tiles: AcrossCounts,
        launch_tiles: usize,
        planes: MatmulPlaneLayout,
    ) -> Self {
        match dispatch {
            UnitDispatch::Sweeping { cores } => Self::in_turns(cores, launch_tiles, planes),
            UnitDispatch::Spread => Self::Boxed(tiles.spread_split(planes.along_m.get())),
        }
    }

    /// A unit for every core of `cores`, within the planes of a cube and no more than the
    /// `launch_tiles`, so that none is launched idle.
    fn in_turns(cores: usize, launch_tiles: usize, planes: MatmulPlaneLayout) -> Self {
        Self::InTurns {
            units: cores.min(planes.count()).min(launch_tiles).max(1),
        }
    }

    /// The order the launch's planes take the tiles of `roles`' plane axes in, a box's taken in
    /// `cube_order` with `along_n` planes along the columns.
    pub fn order(self, roles: &ConvRoles, along_n: usize, cube_order: CubeOrder) -> PlaneOrder {
        match self {
            Self::Boxed(split) => {
                let axes = roles.axes();
                PlaneOrder::new(
                    &[
                        (axes.m, split.rows),
                        (axes.n, along_n),
                        (roles.free.across, split.across),
                    ],
                    cube_order,
                )
            }
            Self::InTurns { units } => PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(units),
        }
    }
}

/// Units each owning a block of the output in registers, reading the gathered input and the rhs
/// straight from global memory through every contracted cell. It asks nothing of the hardware: no
/// shared stage, no plane instruction, so it serves a device with neither.
///
/// An operand whose buffer does not run along the axis a register block reads it in is read one
/// element at a time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitRegistersConvRoutine;

impl UnitRegistersConvRoutine {
    /// How a cube laid out as `planes` takes the tiles of `microkernel`'s blocks, `blocks` along
    /// the columns to a plane, as `hw` runs them.
    fn plane_deal(
        hw: &Hardware,
        problem: &ConvProblem,
        microkernel: &UnitRegistersMicrokernel,
        blocks: usize,
        planes: MatmulPlaneLayout,
    ) -> PlaneDeal {
        let roles = problem.roles();
        let axes = roles.axes();
        let tiles = AcrossCounts {
            rows: problem.extent(axes.m).div_ceil(microkernel.rows),
            across: problem.extent(roles.free.across),
        };
        // The resolved rows, which a work item holding its rows whole sets apart from the settings'.
        let columns = microkernel.settings.columns.get() * blocks;
        let launch_tiles = tiles.count()
            * problem.extent(axes.n).div_ceil(columns)
            * problem.extent(roles.free.batch);
        PlaneDeal::of(hw.unit_dispatch(), tiles, launch_tiles, planes)
    }

    /// Where the memory leaf reads the input, `width` elements to a vector, in a block of the
    /// output.
    ///
    /// # Errors
    ///
    /// The leaf's refusal, where it cannot serve the vectors as they are.
    pub fn check_lhs_vector_axis(
        problem: &ConvProblem,
        width: usize,
    ) -> Result<LhsVectorAxis, Refusal> {
        let roles = problem.roles();
        let space = problem.space();
        LhsVectorAxis::of(
            &space.subspace(&roles.lhs),
            &space.subspace(&roles.rhs),
            &space.subspace(&roles.free.in_order()),
            width,
        )
    }

    /// The axes the planes are laid over, and the extent of each as the launch reads it: a plane
    /// past the end of one has nothing to do.
    fn bounds(problem: &ConvProblem) -> (Vec<Axis>, SequenceArg<usize>) {
        let axes = problem.roles().plane_axes().to_vec();
        let mut extents = SequenceArg::new();
        for &axis in &axes {
            extents.push(problem.extent(axis));
        }
        (axes, extents)
    }

    /// What the block's accesses require of every operand's width, by the axis the operand runs
    /// along: down the contraction a vector of the block's depth, down the rows or the columns a
    /// vector of them, along neither one element at a time. A rhs running down the contraction is
    /// vectorized together with the input, one along the columns together with the output where
    /// that runs along them too.
    fn vectorization(
        &self,
        problem: &ConvProblem,
        microkernel: &UnitRegistersMicrokernel,
        partitioning: &ConvPartitioning,
    ) -> Vectorization<MatmulOperand> {
        let mut vectorization = Vectorization::new();
        self.vectorize(microkernel, partitioning, &mut vectorization);
        let roles = problem.roles();
        let axes = roles.axes();
        let runs_along = |axis| axes.dim(axis);
        if runs_along(roles.inner.lhs) == Some(MatmulDim::K) {
            vectorization.require(VectorRequirement::Access {
                operand: MatmulOperand::Lhs,
                elements: problem.extent(axes.k),
            });
        }
        let out_beside_rhs = [roles.inner.rhs, roles.inner.out]
            .into_iter()
            .all(|axis| runs_along(axis) == Some(MatmulDim::N));
        match out_beside_rhs {
            true => {
                vectorization.require(VectorRequirement::Access {
                    operand: MatmulOperand::Out,
                    elements: microkernel.settings.columns.get(),
                });
                vectorization.require(VectorRequirement::Together {
                    operand: MatmulOperand::Rhs,
                    with: MatmulOperand::Out,
                });
            }
            false => vectorization.require(VectorRequirement::Elementwise {
                operand: MatmulOperand::Out,
            }),
        }
        if runs_along(roles.inner.rhs) == Some(MatmulDim::K) {
            vectorization.require(VectorRequirement::Together {
                operand: MatmulOperand::Rhs,
                with: MatmulOperand::Lhs,
            });
        }
        vectorization
    }
}

impl Routine for UnitRegistersConvRoutine {
    type Problem = ConvProblem;
    type Microkernel = UnitRegistersMicrokernel;
    type Settings = UnitRegistersConvSettings;
    type Blueprint = TileConvBlueprint<UnitRegistersMicrokernel>;
    type Partitioning = ConvPartitioning;

    /// The largest block of the problem's form, then one cell, on the planes, blocks per plane and
    /// cube order the matmul states for a CPU runtime
    /// ([`UnitRegistersMatmulRoutineSettings::cpu_col_rhs`]), which this floor takes on every
    /// device.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let base = UnitRegistersMatmulRoutineSettings::cpu_col_rhs(hw);
        [UnitBlock::Largest, UnitBlock::OneCell]
            .map(|block| Self::Settings {
                block,
                planes: base.planes,
                blocks_per_plane: base.blocks_per_plane,
                cube_order: base.cube_order,
            })
            .to_vec()
    }

    /// The runs the input and the rhs are read in along their contiguous axes: the rest of a
    /// block's accesses depend on the problem, which [`Self::vectorization`] states.
    fn vectorize(
        &self,
        _: &UnitRegistersMicrokernel,
        partitioning: &ConvPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        partitioning.vectorize(vectorization);
    }

    /// What one unit's box moves and computes: its rows of the input and its columns of the rhs
    /// over every contracted cell, read from global memory, and its block written once.
    fn estimate(
        &self,
        problem: &ConvProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        estimate.staging = Some(
            blueprint
                .partitioning
                .staging(&problem.elements, StageRing::SINGLE),
        );
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// The register block `settings` size, a plane of the launch for every box of the output laid
    /// over the rows and the across axis, its units taking blocks along the columns in turns. The
    /// plane is as wide as the device dispatches, since units cooperate on nothing.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let roles = problem.roles();
        let (rows, columns) = BlockForm::of(&roles).size(settings.block, hw, problem.sum);
        let along = |count: usize| NonZeroUsize::new(count).unwrap_or(NonZeroUsize::MIN);
        let block = UnitRegistersMatmulRoutineSettings {
            microkernel: UnitRegistersMicrokernelSettings {
                rows: along(rows),
                columns: along(columns),
                depth: along(BlockDepth::of(hw).size(hw, problem)),
            },
            planes: settings.planes,
            blocks_per_plane: settings.blocks_per_plane,
            cube_order: settings.cube_order,
        };
        let microkernel =
            UnitRegistersMicrokernel::best(hw, &problem.work_item(), &block.microkernel)?;
        let axes = roles.axes();
        let planes = settings.planes;
        let plane_width = UnitRegistersMatmulRoutine::plane_width(hw, planes.count());
        let blocks = block.fitted_blocks_per_plane(problem.extent(axes.n), plane_width);
        let order = Self::plane_deal(hw, problem, &microkernel, blocks, planes).order(
            &roles,
            planes.along_n.get(),
            settings.cube_order,
        );
        let partitioning = ConvPartitioning::unit_registers(problem, &microkernel, blocks, order);
        let vectorization = self.vectorization(problem, &microkernel, &partitioning);
        let cube = CubeResources {
            plane_width,
            shared_memory: 0,
        };
        let build = |vectorization| {
            TileConvBlueprint::new(
                hw,
                problem,
                microkernel,
                partitioning.clone(),
                vectorization,
                ConvStaging::Direct,
                cube,
            )
        };
        let (blueprint, grid) = build(vectorization.clone())?;
        if Self::check_lhs_vector_axis(problem, blueprint.widths.lhs).is_ok() {
            return Ok((blueprint, grid));
        }
        // The leaf serves a one-wide input in every block, so the setting never declines here.
        let mut narrowed = vectorization;
        narrowed.require(VectorRequirement::Elementwise {
            operand: MatmulOperand::Lhs,
        });
        build(narrowed)
    }
}

impl ConvRoutine<UnitRegistersMicrokernel> for UnitRegistersConvRoutine {
    /// Every plane of the launch, its units each on a block of the output.
    fn dispatch(
        &self,
        problem: &ConvProblem,
        binding: ConvBinding,
        _: TileConvBlueprint<UnitRegistersMicrokernel>,
    ) {
        let ConvBinding {
            launcher,
            input,
            rhs,
            out,
        } = binding;
        let (bounded, extents) = Self::bounds(problem);
        // SAFETY: every operand is a tile bound through the launcher, which masks the accesses a
        // box reaches past its tensor; the kernel indexes no buffer but through those tiles and
        // holds no shared memory.
        unsafe {
            unit_registers_conv_kernel::launch_unchecked(
                launcher.client(),
                launcher.cube_count(),
                launcher.cube_dim(),
                input,
                rhs,
                out,
                launcher.partitioning_arg(),
                UnitRegistersMicrokernel::register_block().split_edge(),
                bounded,
                extents,
                problem.kernel_elements(),
            );
        }
    }
}
