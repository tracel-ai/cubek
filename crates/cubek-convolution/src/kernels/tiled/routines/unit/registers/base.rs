use super::kernel::unit_registers_conv_kernel;
use crate::kernels::tiled::{
    ConvBinding, ConvBlueprint, ConvPartitioning, ConvProblem, ConvRoutine,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Axis;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::ops::matmul::LhsVectorAxis;
use cubecl::tile::space::PlaneOrder;
use cubek_matmul::kernels::{UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings};
use cubek_microkernel::matmul::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// What a launch is declined with where the input's lines do not divide the contracted axis: the
/// leaf walks the contraction in whole lines, and a line past the axis's end would read another's.
pub const CONV_UNIT_LINES_UNDIVIDED: Refusal = Refusal::Declined {
    why: "a unit block walks the contraction in whole lines of the input, which must divide the contracted axis",
};

/// Units each owning a block of the output in registers, reading the gathered input and the rhs
/// straight from global memory through every contracted cell. It asks nothing of the hardware: no
/// shared stage, no plane instruction, so it serves a device with neither.
///
/// An operand whose buffer does not run along the axis a register block reads it in is read one
/// element at a time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitRegistersConvRoutine;

impl UnitRegistersConvRoutine {
    /// The routine.
    pub const fn new() -> Self {
        Self
    }

    /// Blocks along an axis the list tries at most.
    const MOST_ALONG: usize = 4;

    /// Vector registers a block's lines may take, of the sixteen a CPU names: the rest is left to
    /// addressing and the compiler's temporaries.
    const LINE_REGISTERS: usize = 12;

    /// The rows and columns of the largest block whose lines fit [`LINE_REGISTERS`]: a cell holds
    /// one line of partial sums, and a step reads one line for each row and each column.
    ///
    /// [`LINE_REGISTERS`]: Self::LINE_REGISTERS
    fn lead_block() -> (usize, usize) {
        let lines = |rows: usize, columns: usize| rows * columns + rows + columns;
        (1..=Self::MOST_ALONG)
            .flat_map(|rows| (1..=Self::MOST_ALONG).map(move |columns| (rows, columns)))
            .filter(|&(rows, columns)| lines(rows, columns) <= Self::LINE_REGISTERS)
            .max_by_key(|&(rows, columns)| (rows * columns, rows))
            .unwrap_or((1, 1))
    }

    /// The planes of a cube along the rows and along the across axis, of at most `planes`, that
    /// cover `tiles` (rows, across) in the fewest blocks. Of those it takes the fewest planes, then
    /// the most along the rows, whose tiles are contiguous.
    pub(super) fn plane_split(tiles: (usize, usize), planes: usize) -> (usize, usize) {
        let blocks =
            |&(rows, across): &(usize, usize)| tiles.0.div_ceil(rows) * tiles.1.div_ceil(across);
        (1..=planes)
            .flat_map(|rows| (1..=planes / rows).map(move |across| (rows, across)))
            .min_by_key(|split| (blocks(split), split.0 * split.1, std::cmp::Reverse(split.0)))
            .unwrap_or((1, 1))
    }

    /// Whether the input's lines, where it is read in them down the contraction, divide the
    /// contracted axis.
    ///
    /// # Errors
    ///
    /// [`CONV_UNIT_LINES_UNDIVIDED`].
    pub fn check_lines(
        problem: &ConvProblem,
        widths: &MatmulOperands<usize>,
    ) -> Result<(), Refusal> {
        let roles = problem.operation.roles();
        let axes = roles.axes();
        let divides =
            roles.inner.lhs != axes.k || problem.extent(axes.k).is_multiple_of(widths.lhs.max(1));
        match divides {
            true => Ok(()),
            false => Err(CONV_UNIT_LINES_UNDIVIDED),
        }
    }

    /// Where the memory leaf reads the input, lined `width` wide, in a block of the output.
    ///
    /// # Errors
    ///
    /// The leaf's refusal, where it cannot serve the lines as they are.
    pub fn check_lhs_vector_axis(
        problem: &ConvProblem,
        width: usize,
    ) -> Result<LhsVectorAxis, Refusal> {
        let roles = problem.operation.roles();
        let space = problem.space();
        LhsVectorAxis::of(
            &space.subspace(&roles.lhs),
            &space.subspace(&roles.rhs),
            &space.subspace(&roles.free),
            width,
        )
    }

    /// The axes a plane may land past the end of, with their extents.
    fn bounds(problem: &ConvProblem) -> Vec<(Axis, usize)> {
        let roles = problem.operation.roles();
        let axes = roles.axes();
        [axes.m, roles.across(), roles.batch()]
            .into_iter()
            .map(|axis| (axis, problem.extent(axis)))
            .collect()
    }

    /// The operands a register block cannot read in lines along the axis it reads them: the lhs
    /// is read down the contraction, the rhs along it for a col rhs and along the columns for a
    /// row one, the output along the columns beside a row rhs.
    fn unlined(problem: &ConvProblem) -> Vec<MatmulOperand> {
        let roles = problem.operation.roles();
        let axes = roles.axes();
        let rhs_read = [axes.k, axes.n].contains(&roles.inner.rhs);
        [
            (MatmulOperand::Lhs, roles.inner.lhs != axes.k),
            (MatmulOperand::Rhs, !rhs_read),
            (
                MatmulOperand::Out,
                roles.inner.rhs == axes.n && roles.inner.out != axes.n,
            ),
        ]
        .into_iter()
        .filter_map(|(operand, unlined)| unlined.then_some(operand))
        .collect()
    }
}

impl Routine for UnitRegistersConvRoutine {
    type Problem = ConvProblem;
    type Microkernel = UnitRegistersMicrokernel;
    type Settings = UnitRegistersMatmulRoutineSettings;
    type Blueprint = ConvBlueprint<UnitRegistersMicrokernel>;
    type Partitioning = ConvPartitioning;

    /// The largest block the registers hold, then one cell, each as deep as the device reads
    /// `f32` in one line, on the plane layout and cube order the matmul's CPU settings take.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let base = UnitRegistersMatmulRoutineSettings::cpu_col_rhs(hw);
        let depth = hw.vector_ceiling(ElemType::Float(FloatKind::F32));
        let along = |count: usize| NonZeroUsize::new(count).unwrap_or(NonZeroUsize::MIN);
        [Self::lead_block(), (1, 1)]
            .into_iter()
            .map(|(rows, columns)| Self::Settings {
                microkernel: UnitRegistersMicrokernelSettings {
                    rows: along(rows),
                    columns: along(columns),
                    depth: along(depth),
                },
                ..base
            })
            .collect()
    }

    /// Nothing of its own: its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &UnitRegistersMicrokernel,
        _: &ConvPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
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
        estimate.staging = Some(blueprint.partitioning.staging(&problem.elements));
        blueprint.microkernel.estimate(
            &problem.registers_work_item(),
            &(),
            grid.planes(),
            estimate,
        );
    }

    /// The register block `settings` size, a plane of the launch for every box of the output, laid
    /// over the rows and the across axis, its units taking blocks along the columns in turns: as wide a plane as the device dispatches,
    /// since units cooperate on nothing.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel = UnitRegistersMicrokernel::best(
            hw,
            &problem.registers_work_item(),
            &settings.microkernel,
        )?;
        let axes = problem.operation.roles().axes();
        let planes = settings.planes;
        let plane_width = UnitRegistersMatmulRoutine::plane_width(hw, planes.count());
        let blocks = UnitRegistersMatmulRoutine::fitted_blocks_per_plane(
            settings.blocks_per_plane.get(),
            problem.extent(axes.n),
            settings.microkernel.columns.get(),
            plane_width,
        );
        let across = problem.operation.roles().across();
        let (along_m, along_across) = Self::plane_split(
            (
                problem.extent(axes.m).div_ceil(microkernel.rows),
                problem.extent(across),
            ),
            planes.along_m.get(),
        );
        let order = PlaneOrder::new(
            &[
                (axes.m, along_m),
                (axes.n, planes.along_n.get()),
                (across, along_across),
            ],
            settings.cube_order,
        );
        let partitioning = ConvPartitioning::units(problem, &microkernel, blocks, order);
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        for operand in Self::unlined(problem) {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
        let roles = problem.operation.roles();
        if roles.inner.lhs == axes.k {
            vectorization.require(VectorRequirement::Access {
                operand: MatmulOperand::Lhs,
                elements: problem.extent(axes.k),
            });
        }
        let (blueprint, grid) = ConvBlueprint::prepared_units(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            plane_width,
        )?;
        Self::check_lines(problem, &blueprint.widths)?;
        Self::check_lhs_vector_axis(problem, blueprint.widths.lhs)?;
        Ok((blueprint, grid))
    }
}

impl ConvRoutine<UnitRegistersMicrokernel> for UnitRegistersConvRoutine {
    /// Every plane of the launch, its units each on a block of the output.
    fn dispatch(
        &self,
        problem: &ConvProblem,
        binding: ConvBinding,
        blueprint: ConvBlueprint<UnitRegistersMicrokernel>,
    ) {
        let ConvBinding {
            launcher,
            input,
            rhs,
            out,
        } = binding;
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
                blueprint.microkernel.register_block().split_edge(),
                Self::bounds(problem),
                problem.kernel_elements(),
            );
        }
    }
}
