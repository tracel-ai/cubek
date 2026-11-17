use super::kernel::hypercube_atomic_reduce_kernel;
use crate::kernels::routines::cube::slot_bytes;
use crate::kernels::{
    ReduceBinding, ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors,
    reduce_widths,
};
use cubecl::client::Client;
use cubecl::ir::Type;
use cubecl::ir::features::AtomicUsage;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{
    AccumulateArgLaunch, Bound, Refusal, StoredTileOperand, TileOperand, Unbound, Vectorization,
};
use cubecl::tile::{Geometry, Levels, Partitioning};
use cubek_microkernel::reduce::cube::{CubeReduceMicrokernel, CubeReduceMicrokernelSettings};
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperand, ReduceOperation};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row and adds its total into the output: a sum of few rows, too long for one
/// cube each to keep the device busy, as a whole tensor summed into one value.
///
/// The output must hold zero before it runs ([`UnitZeroRoutine`](crate::kernels::UnitZeroRoutine)
/// zeroes it); a caller never binds it, since it only ever adds. Its cubes run along the reduced
/// dim and at most two row dims; it declines more.
#[derive(Debug, Clone, Copy, Default)]
pub struct HypercubeAtomicReduceRoutine;

/// The cube microkernel's planes, and how many lines each unit reads of a part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAtomicReduceRoutineSettings {
    pub microkernel: CubeReduceMicrokernelSettings,
    pub unit_lines: usize,
}

impl HypercubeAtomicReduceRoutineSettings {
    /// Lines each unit reads of a part, the longest parts first: the fewest atomic adds that still
    /// spread a row over the device.
    pub const UNIT_LINES: [usize; 3] = [32, 8, 2];
}

impl core::fmt::Display for HypercubeAtomicReduceRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} unit_lines={}", self.microkernel, self.unit_lines)
    }
}

pub type HypercubeAtomicReduceRoutineBlueprint = ReduceRoutineBlueprint<CubeReduceMicrokernel>;

/// Most row dims its cubes run along, beside the reduced dim: a launch grid has three.
const MAX_ROW_DIMS: usize = 2;

/// Why it declines a caller-bound output: it only adds into a buffer it holds at zero.
pub const ATOMIC_OUTPUT_UNBOUND: Refusal = Refusal::Declined {
    why: "the atomic hypercube adds into its output, which a caller cannot bind",
};

impl Routine for HypercubeAtomicReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = CubeReduceMicrokernel;
    type Settings = HypercubeAtomicReduceRoutineSettings;
    type Blueprint = HypercubeAtomicReduceRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes, each with the longest parts first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                HypercubeAtomicReduceRoutineSettings::UNIT_LINES.map(|unit_lines| {
                    HypercubeAtomicReduceRoutineSettings {
                        microkernel,
                        unit_lines,
                    }
                })
            })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &CubeReduceMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ReduceOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if problem.operation != ReduceOperation::Sum {
            return Err(Refusal::Declined {
                why: "the atomic hypercube adds its parts, which only a sum's parts make whole",
            });
        }
        if !hw
            .properties
            .atomic_type_usage(Type::atomic(problem.elements.out))
            .contains(AtomicUsage::Add)
        {
            return Err(Refusal::Declined {
                why: "the device has no atomic add for the output's element",
            });
        }
        if problem.rows().len() > MAX_ROW_DIMS {
            return Err(Refusal::Declined {
                why: "the atomic hypercube runs its cubes along at most two row dims",
            });
        }
        let microkernel =
            CubeReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widths = reduce_widths(hw, problem, vectorization);
        let units = microkernel.lanes * microkernel.planes;
        // Along, a line holds `width` values of the row; across, one.
        let part = units
            * settings.unit_lines
            * match problem.layout() {
                ReduceLayout::Along => widths[0],
                ReduceLayout::Across => 1,
            };
        let reduced = problem.reduced();
        let owner = [problem.owner_box(widths[0]).as_slice(), &[(reduced, part)]].concat();
        // The parts ride the grid's `X`, the boxes of rows its `Y` and `Z`.
        let cube_axes = [&[reduced], problem.rows().as_slice()].concat();
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&owner).cubes(&cube_axes).build(),
        );
        let cube = CubeResources {
            plane_width: units as u32,
            shared_memory: microkernel.planes * slot_bytes(problem, widths[0]),
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceRoutineBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
        &self,
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }
}

impl ReduceRoutine<CubeReduceMicrokernel> for HypercubeAtomicReduceRoutine {
    /// Never reached: [`launch`](Self::launch) binds the output to add into, and
    /// [`launch_unbound`](Self::launch_unbound) declines.
    fn dispatch<I: TileOperand, O: TileOperand>(
        &self,
        _problem: &ReduceRoutineProblem,
        _client: &Client,
        _binding: ReduceBinding<
            <I::Arg as LaunchArg>::RuntimeArg,
            <O::Arg as LaunchArg>::RuntimeArg,
        >,
        _blueprint: HypercubeAtomicReduceRoutineBlueprint,
        _grid: Grid,
    ) {
        unreachable!("{ATOMIC_OUTPUT_UNBOUND:?}")
    }

    /// `tensors` reduced on `grid`, each cube's total added into the output.
    fn launch(
        &self,
        tensors: &ReduceTensors,
        blueprint: HypercubeAtomicReduceRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let elements = problem.elements;
        let binding = tensors
            .bind(&blueprint, &grid)
            .map(Bound::arg, |out: Bound| {
                let (spec, width) = (out.spec.clone(), out.vector_size);
                AccumulateArgLaunch::new(out.tensor(), width, spec)
            });
        // SAFETY: both operands are bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; the output only receives atomic adds. Its shared memory is the slots its
        // microkernel declares, which the grid counted.
        unsafe {
            hypercube_atomic_reduce_kernel::launch_unchecked::<StoredTileOperand>(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.input,
                binding.out,
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                [elements.input, elements.accumulator, elements.out],
            )
        }
    }

    fn launch_unbound<I: TileOperand, O: TileOperand>(
        &self,
        _tensors: &ReduceTensors<Geometry>,
        _blueprint: HypercubeAtomicReduceRoutineBlueprint,
        _grid: Grid,
        _input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
        _out: impl FnOnce(Unbound) -> <O::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        Err(ATOMIC_OUTPUT_UNBOUND)
    }
}
