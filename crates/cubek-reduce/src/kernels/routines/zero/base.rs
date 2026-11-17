use super::kernel::unit_zero_kernel;
use crate::kernels::{
    ReduceBinding, ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors,
    reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, TileOperand, Unbound, Vectorization};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Geometry, Launcher, Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::zero::ZeroOperand;
use cubek_microkernel::zero::unit::{UnitZeroMicrokernel, UnitZeroMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// A reduction's output zeroed, a unit per box of rows' cells: what a reduction that adds its
/// parts into the output starts from ([`HypercubeAtomicReduceRoutine`](crate::kernels::HypercubeAtomicReduceRoutine)).
#[derive(Debug, Clone, Copy, Default)]
pub struct UnitZeroRoutine;

/// The zero microkernel's setting, and how many planes each cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitZeroRoutineSettings {
    pub microkernel: UnitZeroMicrokernelSettings,
    pub planes: usize,
}

impl core::fmt::Display for UnitZeroRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.planes)
    }
}

pub type UnitZeroRoutineBlueprint = ReduceRoutineBlueprint<UnitZeroMicrokernel>;

impl Routine for UnitZeroRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitZeroMicrokernel;
    type Settings = UnitZeroRoutineSettings;
    type Blueprint = UnitZeroRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Cubes of four planes: a zeroing is one write a unit, which no setting speeds up.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        UnitZeroMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| UnitZeroRoutineSettings {
                microkernel,
                planes: 4,
            })
            .collect()
    }

    fn vectorize(
        &self,
        microkernel: &UnitZeroMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ZeroOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Placed as the unit reduce places its owners, so each unit zeroes the cells it would write.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel = UnitZeroMicrokernel::best(hw, &(), &settings.microkernel)?;
        let widths = reduce_widths(hw, problem, Vectorization::new());
        let plane_width = hw.dispatch_plane_size() as usize;
        let mut owners = problem.owner_axes();
        owners.reverse();
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box(widths[0]))
                .launch_units(
                    &owners,
                    UnitCube {
                        plane_width,
                        planes: settings.planes,
                    },
                )
                .build(),
        );
        let cube = CubeResources {
            plane_width: plane_width as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceRoutineBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
        &self,
        _problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&(), &(), grid.planes(), estimate);
    }
}

impl ReduceRoutine<UnitZeroMicrokernel> for UnitZeroRoutine {
    /// The output zeroed; the input is never read.
    fn dispatch<I: TileOperand>(
        &self,
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        _blueprint: UnitZeroRoutineBlueprint,
        grid: Grid,
    ) {
        launch_zero(client, problem, binding.out, &binding.launcher, grid);
    }

    /// The output zeroed. The caller's input is never read, so `input` is never called.
    fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        blueprint: UnitZeroRoutineBlueprint,
        grid: Grid,
        _input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        self.zero_unbound(tensors, blueprint, grid);
        Ok(())
    }
}

impl UnitZeroRoutine {
    /// The output of `tensors`, whose input the caller binds, zeroed on `grid`.
    pub fn zero_unbound(
        &self,
        tensors: &ReduceTensors<Geometry>,
        blueprint: UnitZeroRoutineBlueprint,
        grid: Grid,
    ) {
        let binding = tensors.bind(&blueprint, &grid);
        launch_zero(
            tensors.client(),
            tensors.problem(),
            binding.out.arg(),
            &binding.launcher,
            grid,
        );
    }
}

/// `out` zeroed on `grid`, through `launcher`'s partitioning.
fn launch_zero(
    client: &Client,
    problem: &ReduceRoutineProblem,
    out: TileTensorLaunch,
    launcher: &Launcher,
    grid: Grid,
) {
    // SAFETY: the output is bound through the launcher, which masks the writes of a tile the
    // problem's edge cuts short, and the kernel indexes no buffer but through its tile; it holds
    // no shared memory.
    unsafe {
        unit_zero_kernel::launch_unchecked(
            client,
            grid.launch.cube_count,
            grid.launch.cube_dim,
            out,
            launcher.partitioning_arg(),
            problem.elements.out,
        )
    }
}
