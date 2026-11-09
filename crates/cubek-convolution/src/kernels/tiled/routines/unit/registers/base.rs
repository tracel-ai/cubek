use super::kernel::unit_registers_conv_kernel;
use crate::kernels::tiled::{
    ConvBinding, ConvBlueprint, ConvOperation, ConvPartitioning, ConvProblem, ConvRoutine,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::RegisterBlock;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
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

/// What a launch is declined with for the weight backward: its output runs along the rows, which
/// the memory leaf a unit block contracts through read back wrong on a CPU runtime, every cell.
pub const CONV_UNIT_WEIGHT_BACKWARD: Refusal = Refusal::Declined {
    why: "the unit registers routine's memory leaf computes a weight backward wrongly, so it is not served",
};

/// Units each owning a block of the output in registers, reading the gathered input and the rhs
/// straight from global memory through every contracted cell. It asks nothing of the hardware: no
/// shared stage, no plane instruction, so it serves a device with neither.
///
/// An operand whose buffer does not run along the axis a register block reads it in is read one
/// element at a time. The weight backward is declined ([`CONV_UNIT_WEIGHT_BACKWARD`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitRegistersConvRoutine {
    weight_backward: bool,
}

impl UnitRegistersConvRoutine {
    /// The routine as served: it declines the weight backward.
    pub const fn new() -> Self {
        Self {
            weight_backward: false,
        }
    }

    /// The routine with the weight backward let through, for the tests that probe why it reads
    /// back wrong.
    #[cfg(feature = "benchmarks")]
    pub const fn probing_weight_backward() -> Self {
        Self {
            weight_backward: true,
        }
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

    /// The block a unit accumulates in: its edge split from its interior, and where the input is
    /// lined along a contraction that steps one value at a time, a line read once for every
    /// component rather than one line per value.
    fn register_block(
        problem: &ConvProblem,
        blueprint: &ConvBlueprint<UnitRegistersMicrokernel>,
    ) -> RegisterBlock {
        let roles = problem.operation.roles();
        let one_value_a_step = roles.inner.rhs != roles.axes().k;
        let block = blueprint.microkernel.register_block().split_edge();
        match one_value_a_step && blueprint.widths.lhs > 1 {
            true => block.component_fanout(),
            false => block,
        }
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

    /// The register block `settings` size, a plane of the launch for every box of the output, its
    /// units taking blocks along the columns in turns: as wide a plane as the device dispatches,
    /// since units cooperate on nothing.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if problem.operation == ConvOperation::BackwardWeight && !self.weight_backward {
            return Err(CONV_UNIT_WEIGHT_BACKWARD);
        }
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
        let order = PlaneOrder::new(
            &[
                (axes.m, planes.along_m.get()),
                (axes.n, planes.along_n.get()),
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
                Self::register_block(problem, &blueprint),
                problem.kernel_elements(),
            );
        }
    }
}
