use super::kernel::cube_max_pool_kernel;
use crate::kernels::geometry::TileGeometry;
use crate::kernels::space;
use crate::kernels::{PoolRoutine, PoolRoutineProblem, PoolTensors};
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::pool::PoolOperand;
use cubek_microkernel::pool::cube::{CubeMaxPoolMicrokernel, CubeMaxPoolMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every cube a box of output rows and columns, walking its channels a block at a time, each
/// block's windows read from the input where it lies and reduced to their largest tap.
///
/// Every device runs it: it holds no shared memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeMaxPoolRoutine;

/// How a cube covers its box: its planes, each walking rows, each unit taking columns and a run
/// of channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeMaxPoolRoutineSettings {
    pub microkernel: CubeMaxPoolMicrokernelSettings,
    pub planes: usize,
    pub rows_per_plane: usize,
    pub cols_per_unit: usize,
}

impl CubeMaxPoolRoutineSettings {
    /// `planes` a cube, each walking `rows_per_plane` rows, each unit `cols_per_unit` columns.
    pub const fn new(planes: usize, rows_per_plane: usize, cols_per_unit: usize) -> Self {
        Self {
            microkernel: CubeMaxPoolMicrokernelSettings,
            planes,
            rows_per_plane,
            cols_per_unit,
        }
    }
}

impl core::fmt::Display for CubeMaxPoolRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} planes={} rows={} cols={}",
            self.microkernel, self.planes, self.rows_per_plane, self.cols_per_unit
        )
    }
}

/// What a launch of [`CubeMaxPoolRoutine`] compiles.
#[derive(Debug, Clone, PartialEq)]
pub struct CubeMaxPoolRoutineBlueprint {
    pub microkernel: CubeMaxPoolMicrokernel,
    pub partitioning: Partitioning,
    /// The width both operands bind at.
    pub width: usize,
    /// Whether some tap reads past the input, so the gather masks its reads.
    pub checked: bool,
}

impl Routine for CubeMaxPoolRoutine {
    type Problem = PoolRoutineProblem;
    type Microkernel = CubeMaxPoolMicrokernel;
    type Settings = CubeMaxPoolRoutineSettings;
    type Blueprint = CubeMaxPoolRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Four planes of two rows a cube by default, then eight and sixteen planes.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeMaxPoolMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [4, 8, 16].map(|planes| CubeMaxPoolRoutineSettings {
                    microkernel,
                    ..CubeMaxPoolRoutineSettings::new(planes, 2, 1)
                })
            })
            .collect()
    }

    /// Its blocks hand the microkernel a unit's channel run: the operands' lines divide it.
    fn vectorize(
        &self,
        microkernel: &CubeMaxPoolMicrokernel,
        partitioning: &Partitioning,
        vectorization: &mut Vectorization<PoolOperand>,
    ) {
        let channels = partitioning.leaf().extent(space::CHANNEL);
        PoolOperand::require_along_channels(channels, vectorization);
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &PoolRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let plane_width = hw.dispatch_plane_size() as usize;
        let geometry = TileGeometry::new(settings, problem.channels, plane_width);
        let partitioning = problem.space(geometry, plane_width).partitioning();
        let microkernel = CubeMaxPoolMicrokernel::best(
            hw,
            &space::work_item(&partitioning),
            &settings.microkernel,
        )?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let width = problem.width(hw, &partitioning, vectorization);
        let cube = CubeResources {
            plane_width: plane_width as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        let blueprint = CubeMaxPoolRoutineBlueprint {
            microkernel,
            partitioning,
            width,
            checked: !problem.taps_in_bounds(),
        };
        Ok((blueprint, grid))
    }

    fn estimate(
        &self,
        _problem: &PoolRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint.microkernel.estimate(
            &space::work_item(&blueprint.partitioning),
            &(),
            grid.planes(),
            estimate,
        );
    }
}

impl PoolRoutine for CubeMaxPoolRoutine {
    fn dispatch(&self, tensors: &PoolTensors, blueprint: CubeMaxPoolRoutineBlueprint, grid: Grid) {
        let problem = tensors.problem();
        let binding = tensors.bind(
            &blueprint.partitioning,
            blueprint.width,
            blueprint.checked,
            &grid,
        );
        cube_max_pool_kernel::launch(
            tensors.client(),
            grid.launch.cube_count,
            grid.launch.cube_dim,
            binding.input.arg(),
            binding.output.arg(),
            binding.launcher.partitioning_arg(),
            problem.element,
        );
    }
}
