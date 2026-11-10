use super::kernel::cube_staged_interpolate_kernel;
use crate::kernels::filter::SeparableFilterFamily;
use crate::kernels::geometry::TileGeometry;
use crate::kernels::space;
use crate::kernels::{
    CubeInterpolateRoutineSettings, InterpolateRoutine, InterpolateRoutineProblem,
    InterpolateTensors,
};
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::interpolate::InterpolateOperand;
use cubek_microkernel::interpolate::cube::CubeInterpolateMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// What [`CubeStagedInterpolateRoutine`] declines with on a CPU, which has no shared memory to
/// stage a window in.
pub const STAGED_INTERPOLATE_ON_CPU: Refusal = Refusal::Declined {
    why: "a CPU has no shared memory to stage the tap window in",
};

/// Every cube a box of output rows and columns, walking its channels a block at a time, each
/// block's tap window staged in shared memory once and read there by every tap.
///
/// It declines where the window outgrows the device's shared memory, and on a CPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeStagedInterpolateRoutine;

/// What a launch of [`CubeStagedInterpolateRoutine`] compiles.
#[derive(Debug, Clone, PartialEq)]
pub struct CubeStagedInterpolateRoutineBlueprint {
    pub microkernel: CubeInterpolateMicrokernel,
    pub partitioning: Partitioning,
    /// The width both operands bind at.
    pub width: usize,
    /// Whether some read of the input is masked: a tap past its edge, or a stage line past its
    /// channels.
    pub checked: bool,
    /// The line width the stage lays a unit's channel run out in, where the tensors bind narrower
    /// than the run: `C = 3` has no 4-aligned row start, but a stage padding the run to 4 does.
    pub padded: Option<usize>,
}

impl Routine for CubeStagedInterpolateRoutine {
    type Problem = InterpolateRoutineProblem;
    type Microkernel = CubeInterpolateMicrokernel;
    type Settings = CubeInterpolateRoutineSettings;
    type Blueprint = CubeStagedInterpolateRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Four planes of two rows a cube by default, then one and two planes.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeInterpolateMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [4, 1, 2].map(|planes| CubeInterpolateRoutineSettings {
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
        if hw.properties.hardware.num_cpu_cores.is_some() {
            return Err(STAGED_INTERPOLATE_ON_CPU);
        }
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
        let run = geometry.channel_block;
        let padded =
            (run != width && run.is_power_of_two() && run <= hw.vector_ceiling(problem.element))
                .then_some(run);
        let shared_memory = space::stage_window_bytes(
            problem.row(),
            problem.col(),
            problem.taps(),
            problem.radius(),
            geometry,
            padded.unwrap_or(width),
            problem.element.size(),
        );
        let cube = CubeResources {
            plane_width: plane_width as u32,
            shared_memory,
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        let blueprint = CubeStagedInterpolateRoutineBlueprint {
            microkernel,
            partitioning,
            width,
            checked: !problem.taps_in_bounds() || padded.is_some(),
            padded,
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

impl InterpolateRoutine for CubeStagedInterpolateRoutine {
    fn dispatch<F: SeparableFilterFamily>(
        &self,
        tensors: &InterpolateTensors,
        blueprint: CubeStagedInterpolateRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let binding = tensors.bind(
            &blueprint.partitioning,
            blueprint.width,
            blueprint.checked,
            &grid,
        );
        cube_staged_interpolate_kernel::launch::<F>(
            tensors.client(),
            grid.launch.cube_count,
            grid.launch.cube_dim,
            binding.input.arg(),
            binding.output.arg(),
            binding.launcher.partitioning_arg(),
            problem.row(),
            problem.col(),
            blueprint.padded,
            blueprint.microkernel,
            problem.element,
        );
    }
}
