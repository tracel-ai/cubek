use super::kernel::cube_interpolate_kernel;
use crate::kernels::filter::SeparableFilterFamily;
use crate::kernels::geometry::TileGeometry;
use crate::kernels::space;
use crate::kernels::{InterpolateRoutine, InterpolateRoutineProblem, InterpolateTensors};
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::interpolate::InterpolateOperand;
use cubek_microkernel::interpolate::cube::{
    CubeInterpolateMicrokernel, CubeInterpolateMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every cube a box of output rows and columns, walking its channels a block at a time, each
/// block's taps read from the input where it lies.
///
/// Every device runs it: it holds no shared memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeInterpolateRoutine;

/// How a cube of the interpolate routines covers its box: its planes, each walking rows, each unit
/// taking columns and a run of channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeInterpolateRoutineSettings {
    pub microkernel: CubeInterpolateMicrokernelSettings,
    pub planes: usize,
    pub rows_per_plane: usize,
    pub cols_per_unit: usize,
    /// The unit's channel run, `None` for the widest line the channels divide. One the channels
    /// do not divide leaves the last block part padding, every read and write in it masked.
    pub channel_block: Option<usize>,
}

impl CubeInterpolateRoutineSettings {
    /// `planes` a cube, each walking `rows_per_plane` rows, each unit `cols_per_unit` columns.
    pub const fn new(planes: usize, rows_per_plane: usize, cols_per_unit: usize) -> Self {
        Self {
            microkernel: CubeInterpolateMicrokernelSettings,
            planes,
            rows_per_plane,
            cols_per_unit,
            channel_block: None,
        }
    }

    /// These settings with the unit's channel run pinned at `block`.
    pub const fn with_channel_block(self, block: usize) -> Self {
        Self {
            channel_block: Some(block),
            ..self
        }
    }
}

impl core::fmt::Display for CubeInterpolateRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} planes={} rows={} cols={}",
            self.microkernel, self.planes, self.rows_per_plane, self.cols_per_unit
        )?;
        match self.channel_block {
            Some(block) => write!(f, " channels={block}"),
            None => Ok(()),
        }
    }
}

/// What a launch of [`CubeInterpolateRoutine`] compiles.
#[derive(Debug, Clone, PartialEq)]
pub struct CubeInterpolateRoutineBlueprint {
    pub microkernel: CubeInterpolateMicrokernel,
    pub partitioning: Partitioning,
    /// The width both operands bind at.
    pub width: usize,
    /// Whether some tap reads past the input, so the gather masks its reads.
    pub checked: bool,
}

impl Routine for CubeInterpolateRoutine {
    type Problem = InterpolateRoutineProblem;
    type Microkernel = CubeInterpolateMicrokernel;
    type Settings = CubeInterpolateRoutineSettings;
    type Blueprint = CubeInterpolateRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Four planes of two rows a cube by default, then eight and sixteen planes.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeInterpolateMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [4, 8, 16].map(|planes| CubeInterpolateRoutineSettings {
                    microkernel,
                    ..CubeInterpolateRoutineSettings::new(planes, 2, 1)
                })
            })
            .collect()
    }

    /// Its blocks hand the microkernel a unit's channel run: the operands' lines divide it.
    fn vectorize(
        &self,
        microkernel: &CubeInterpolateMicrokernel,
        partitioning: &Partitioning,
        vectorization: &mut Vectorization<InterpolateOperand>,
    ) {
        let channels = partitioning.leaf().extent(space::CHANNEL);
        InterpolateOperand::require_along_channels(channels, vectorization);
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &InterpolateRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let plane_width = hw.dispatch_plane_size() as usize;
        let geometry = TileGeometry::new(settings, problem.forward.channels, plane_width);
        let partitioning = problem.space(geometry, plane_width).partitioning();
        let microkernel = CubeInterpolateMicrokernel::best(
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
        let blueprint = CubeInterpolateRoutineBlueprint {
            microkernel,
            partitioning,
            width,
            checked: !problem.taps_in_bounds(),
        };
        Ok((blueprint, grid))
    }

    fn estimate(
        &self,
        _problem: &InterpolateRoutineProblem,
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

impl InterpolateRoutine for CubeInterpolateRoutine {
    fn dispatch<F: SeparableFilterFamily>(
        &self,
        tensors: &InterpolateTensors,
        blueprint: CubeInterpolateRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(
            &blueprint.partitioning,
            blueprint.width,
            blueprint.checked,
            &grid,
        );
        cube_interpolate_kernel::launch::<F>(
            tensors.client(),
            grid.launch.cube_count,
            grid.launch.cube_dim,
            binding.input.arg(),
            binding.output.arg(),
            binding.launcher.partitioning_arg(),
            problem.row(),
            problem.col(),
            blueprint.microkernel,
            problem.element,
        );
    }
}
