use super::base;
use super::kernel::cube_causal_conv_kernel;
use crate::kernels::conv::{
    CausalConvBlueprint, CausalConvInputs, CausalConvOutputs, CausalConvPast, CausalConvProblem,
    CausalConvTensors,
};
use crate::kernels::refusals::admit_on_device;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::linear_recurrence::CausalConvPartitioning;
use cubek_microkernel::linear_recurrence::conv::CausalConvOperand;
use cubek_microkernel::linear_recurrence::conv::cube::{
    CausalConvSettings, CubeCausalConvMicrokernel,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, PreparedLaunches, Routine};

/// The causal conv with a box of channel lines of a row, or of a position of a prompt, in each
/// cube, a line to each unit. It reads no plane feature, so it is the floor every device runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeCausalConvRoutine;

/// The blueprint of [`CubeCausalConvRoutine`].
pub type CubeCausalConvRoutineBlueprint = CausalConvBlueprint<CubeCausalConvMicrokernel>;

impl CubeCausalConvRoutine {
    /// The causal conv of `inputs`, launched on their client's device, prepared for `hw`.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the conv on these inputs.
    pub fn launch_on(
        &self,
        hw: &Hardware,
        inputs: CausalConvInputs,
    ) -> Result<CausalConvOutputs, Refusal> {
        let tensors = CausalConvTensors::new(inputs)?;
        let (blueprint, grid) = self.prepare_first_served(hw, tensors.problem())?;
        self.launch(hw, tensors, blueprint, grid)
    }

    /// [`launch_on`](Self::launch_on), preparing a problem once however often it is launched.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the conv on these inputs.
    pub fn launch_prepared_on(
        &self,
        prepared: &PreparedLaunches<Self>,
        hw: &Hardware,
        inputs: CausalConvInputs,
    ) -> Result<CausalConvOutputs, Refusal> {
        let tensors = CausalConvTensors::new(inputs)?;
        let (blueprint, grid) =
            prepared.prepare(self, hw, tensors.problem(), &CausalConvSettings)?;
        self.launch(hw, tensors, blueprint, grid)
    }

    fn launch(
        &self,
        hw: &Hardware,
        tensors: CausalConvTensors,
        blueprint: CubeCausalConvRoutineBlueprint,
        grid: Grid,
    ) -> Result<CausalConvOutputs, Refusal> {
        let problem = tensors.problem();
        let (element, output_element) = (problem.element, problem.output_element);
        let columns = problem.past.ring().columns;
        let binding = tensors.bind(&blueprint, &grid);
        // SAFETY: every operand is bound through the launcher and indexed only through its tile,
        // whose reads past an edge are masked. A unit writes only the ring channels it reads, and
        // a cube only its own position and box of the output.
        unsafe {
            cube_causal_conv_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count.clone(),
                grid.launch.cube_dim,
                blueprint.width,
                binding.reads.input,
                binding.reads.filters,
                binding.bias.into(),
                binding.reads.past.ring,
                binding.reads.past.lengths,
                binding.output,
                columns,
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                element,
                output_element,
            );
        }
        if let Some(write_back) = &blueprint.write_back {
            write_back.launch(hw, tensors.reads());
        }
        Ok(tensors.into_outputs())
    }
}

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

impl Routine for CubeCausalConvRoutine {
    type Problem = CausalConvProblem;
    type Microkernel = CubeCausalConvMicrokernel;
    type Settings = CausalConvSettings;
    type Blueprint = CubeCausalConvRoutineBlueprint;
    type Partitioning = CausalConvPartitioning;

    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeCausalConvMicrokernel::settings(hw)
    }

    fn vectorize(
        &self,
        microkernel: &CubeCausalConvMicrokernel,
        _: &CausalConvPartitioning,
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
        let stream = |elements: usize, run: usize, element_bytes: usize| GlobalStream {
            bytes: (elements * element_bytes) as u64,
            run_bytes: run * element_bytes,
        };
        let outputs = stream(channels, channels, output_size);
        let filters = stream(channels * (problem.taps + 1), problem.taps, size);
        let (reads, writes) = match problem.past {
            CausalConvPast::Window { .. } => (
                vec![
                    stream(channels, channels, size),
                    filters,
                    stream(channels * (problem.taps - 1), channels, size),
                ],
                vec![stream(channels, channels, size), outputs],
            ),
            CausalConvPast::Prompt { .. } => (
                vec![stream(channels * problem.taps, channels, size), filters],
                vec![outputs],
            ),
        };
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

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &CausalConvProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        base::prepare(hw, problem, settings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::conv::RingStore;
    use crate::kernels::conv::fixtures::{binding_enough, fleet};
    use core::num::NonZeroUsize;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::linear_recurrence::LINE_IN_BOX;
    use cubek_std::devices;
    use cubek_std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn ring(batch: usize, taps: usize) -> RingStore {
        RingStore {
            slots: NonZeroUsize::new(batch + 3).unwrap(),
            columns: taps + 1,
        }
    }

    fn conv(batch: usize, channels: usize, taps: usize) -> CausalConvProblem {
        let past = CausalConvPast::Window {
            ring: ring(batch, taps),
        };
        CausalConvProblem::dense(batch, channels, taps, past, F16, F16)
    }

    fn prompt(batch: usize, positions: usize, channels: usize, taps: usize) -> CausalConvProblem {
        let past = CausalConvPast::Prompt {
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
                let boxes = lines.div_ceil(blueprint.microkernel.units);
                assert_eq!(
                    blueprint.partitioning.as_cubek().distributed(LINE_IN_BOX),
                    Some(blueprint.microkernel.units)
                );
                let cubes = problem.batch * problem.past.positions() * boxes;
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
