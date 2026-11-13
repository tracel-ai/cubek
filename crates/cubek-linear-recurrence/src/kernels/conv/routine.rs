use super::kernel::cube_causal_conv_kernel;
use crate::kernels::conv::{
    CausalConvBinding, CausalConvBlueprint, CausalConvProblem, CausalConvRoutine,
    CausalConvTensors, CausalConvTraffic, RingWriteBackLaunch,
};
use crate::kernels::refusals::admit_on_device;
use cubecl::tile::Levels;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::linear_recurrence::conv::CausalConvOperand;
use cubek_microkernel::linear_recurrence::conv::cube::{
    CausalConvSettings, CubeCausalConvMicrokernel,
};
use cubek_microkernel::linear_recurrence::{
    CHANNEL_RUN, LINE_IN_RUN, POSITION, RecurrencePartitioning,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// The causal conv with a run of channel lines of a position of a row in each cube, a line to each
/// unit. It reads no plane feature, so it is the floor every device runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeCausalConvRoutine;

/// The blueprint of [`CubeCausalConvRoutine`].
pub type CubeCausalConvRoutineBlueprint = CausalConvBlueprint<CubeCausalConvMicrokernel>;

/// Whether `hw` runs `problem`: what a caller asks before it falls back to composed ops.
///
/// # Errors
///
/// [`ELEMENT_NOT_ON_DEVICE`](cubek_matmul::kernels::tensor::ELEMENT_NOT_ON_DEVICE) where the
/// device lacks the activations' element or the output's, or what [`CausalConvProblem::admit`]
/// declines.
pub fn serves_causal_conv(hw: &Hardware, problem: &CausalConvProblem) -> Result<(), Refusal> {
    CausalConvProblem::admit_elements(problem.element, problem.output_element)?;
    admit_on_device(hw, [problem.element, problem.output_element])?;
    problem.admit(hw)
}

/// The widest width the device, the microkernel's accesses and every operand's alignment and view
/// start admit.
fn settled_width(
    hw: &Hardware,
    problem: &CausalConvProblem,
    microkernel: &CubeCausalConvMicrokernel,
) -> usize {
    let element = |operand| match operand {
        CausalConvOperand::Output => problem.output_element,
        _ => problem.element,
    };
    let mut vectorization = Vectorization::new();
    microkernel.vectorize(&(), &mut vectorization);
    let bound = |operand: &CausalConvOperand| problem.bias || *operand != CausalConvOperand::Bias;
    for operand in CausalConvOperand::CHANNELS.iter().filter(|o| bound(o)) {
        let operand = *operand;
        vectorization.require(VectorRequirement::Aligned {
            operand,
            alignment: problem.alignments.get(operand),
        });
        vectorization.require(VectorRequirement::Aligned {
            operand,
            alignment: ElemAlignment::at_start(
                problem.start_bytes.get(operand),
                element(operand).size(),
            ),
        });
    }
    vectorization.width(CausalConvOperand::Input, |operand| {
        hw.vector_ceiling(element(operand))
    })
}

impl Routine for CubeCausalConvRoutine {
    type Problem = CausalConvProblem;
    type Microkernel = CubeCausalConvMicrokernel;
    type Settings = CausalConvSettings;
    type Blueprint = CubeCausalConvRoutineBlueprint;
    type Partitioning = RecurrencePartitioning;

    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeCausalConvMicrokernel::settings(hw)
    }

    fn vectorize(
        &self,
        microkernel: &CubeCausalConvMicrokernel,
        _: &RecurrencePartitioning,
        vectorization: &mut Vectorization<CausalConvOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn estimate(
        &self,
        problem: &CausalConvProblem,
        blueprint: &CubeCausalConvRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let channels = blueprint.microkernel.units * blueprint.width;
        let (size, output_size) = (problem.element.size(), problem.output_element.size());
        let stream = |elements: usize, element_bytes: usize, run: usize| GlobalStream {
            bytes: (elements * element_bytes) as u64,
            run_bytes: run * element_bytes,
        };
        let (reads, writes) = problem.past.streams(&CausalConvTraffic {
            channels,
            taps: problem.taps,
            run: channels,
            element_size: size,
            filters: stream(channels * problem.filter_values(), size, problem.taps),
            outputs: stream(channels, output_size, channels),
        });
        estimate.staging = Some(CubeStaging {
            operations: problem.work_item().operations(channels) as f64,
            reads,
            writes,
            stages: 1,
            stages_in_flight: 1,
            drain_atomics: 0,
        });
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// The launch of `problem` on `hw`: no cube holds more units than the channels have lines, and
    /// the write back of a prompt is checked against the device too.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &CausalConvProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.admit_launch(hw)?;
        let item = problem.work_item();
        let provisional = CubeCausalConvMicrokernel::best(hw, &item, settings)?;
        let width = settled_width(hw, problem, &provisional);
        let lines = problem.channels / width;
        let microkernel = CubeCausalConvMicrokernel {
            units: provisional.units.min(lines),
            ..provisional
        };
        let partitioning = RecurrencePartitioning::conv(
            problem.space(microkernel.units, width),
            Levels::leaf(&problem.leaf(width)).units_distributed(LINE_IN_RUN, microkernel.units),
            &[POSITION, CHANNEL_RUN],
        );
        let cube = CubeResources {
            plane_width: microkernel.units as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
        let write_back =
            RingWriteBackLaunch::prepare(hw, problem, width, microkernel.channel_run())?;
        let blueprint = CausalConvBlueprint::new(microkernel, partitioning, width, write_back);
        Ok((blueprint, grid))
    }
}

impl CausalConvRoutine<CubeCausalConvMicrokernel> for CubeCausalConvRoutine {
    fn dispatch(
        &self,
        tensors: &CausalConvTensors,
        binding: CausalConvBinding,
        blueprint: CubeCausalConvRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        // SAFETY: every operand is bound through the launcher and indexed only through its tile,
        // whose reads past an edge are masked. A unit writes only the ring channels it reads, and
        // a cube only its own position and run of the output.
        unsafe {
            cube_causal_conv_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count.clone(),
                grid.launch.cube_dim,
                blueprint.width,
                binding.input,
                binding.filters,
                binding.bias.into(),
                binding.past.store,
                binding.past.lengths,
                binding.output,
                problem.past.ring().columns,
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                problem.element,
                problem.output_element,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::conv::{CausalConvPastLayout, RingLayout};
    use crate::kernels::fixtures::{binding_enough, fleet};
    use core::num::NonZeroUsize;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::linear_recurrence::conv::cube::RingWriter;
    use cubek_std::devices;
    use cubek_std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn ring(batch: usize, taps: usize) -> RingLayout {
        RingLayout {
            slots: NonZeroUsize::new(batch + 3).unwrap(),
            columns: taps + 1,
        }
    }

    fn conv(batch: usize, channels: usize, taps: usize) -> CausalConvProblem {
        let past = CausalConvPastLayout::DecodeStep {
            ring: ring(batch, taps),
        };
        CausalConvProblem::dense(batch, channels, taps, past, F16, F16)
    }

    fn prompt(batch: usize, positions: usize, channels: usize, taps: usize) -> CausalConvProblem {
        let past = CausalConvPastLayout::Prompt {
            ring: ring(batch, taps),
            positions,
        };
        CausalConvProblem::dense(batch, channels, taps, past, F16, F16)
    }

    fn problems() -> Vec<CausalConvProblem> {
        vec![
            conv(1, 1280, 4),
            conv(8, 6144, 4),
            conv(3, 37 * 5, 4),
            conv(2, 12, 2),
            conv(5, 3, 3),
            prompt(1, 512, 1280, 4),
            prompt(2, 67, 6144, 4),
            prompt(3, 1, 96, 2),
            prompt(1, 9, 7, 3),
        ]
    }

    /// The floor prepares every problem on every device of the fleet, admission is that
    /// preparation, and the launch is the microkernel's: its units no more than the lines, and an
    /// estimate that holds the work.
    #[test]
    fn the_floor_prepares_everything_admission_admits() {
        for hw in fleet() {
            for problem in problems() {
                let prepared = CubeCausalConvRoutine.prepare_first_served(&hw, &problem);
                assert_eq!(problem.admit(&hw), prepared.clone().map(|_| ()));
                let (blueprint, grid) =
                    prepared.unwrap_or_else(|why| panic!("{problem:?}: {why:?}"));
                assert_eq!(grid.shared_memory, 0);
                assert_eq!(
                    grid.launch.cube_dim.num_elems() as usize,
                    blueprint.microkernel.units
                );
                let lines = problem.channels / blueprint.width;
                assert!(blueprint.microkernel.units <= lines);
                assert_eq!(
                    blueprint.write_back.is_some(),
                    problem.past.writer() == RingWriter::WriteBackLaunch
                );
                let runs = lines.div_ceil(blueprint.microkernel.units);
                assert_eq!(
                    blueprint.partitioning.as_cubek().distributed(LINE_IN_RUN),
                    Some(blueprint.microkernel.units)
                );
                let cubes = problem.batch * problem.past.positions() * runs;
                assert_eq!(grid.cubes(), Some(cubes as u64));
                let mut estimate = KernelEstimate::from(&grid);
                CubeCausalConvRoutine.estimate(&problem, &blueprint, &grid, &mut estimate);
                let staging = estimate.staging.as_ref().expect("staging");
                let performed = staging.operations * estimate.cubes.expect("a static grid") as f64;
                assert!(performed >= problem.demand().compute_ops as f64 * 0.5);
            }
        }
    }

    /// The positions of a prompt ride the grid, so admission declines what a device's cube count
    /// cannot hold.
    #[test]
    fn admission_follows_the_grid_a_device_holds() {
        let hw = binding_enough(devices::wgpu_no_subgroups());
        assert_eq!(hw.properties.hardware.max_cube_count.0, 65535);
        assert!(prompt(1, 65535, 16, 4).admit(&hw).is_ok());
        assert!(prompt(1, 70000, 16, 4).admit(&hw).is_err());
    }
}
