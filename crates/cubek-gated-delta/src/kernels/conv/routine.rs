use super::kernel::cube_gated_delta_conv_kernel;
use crate::kernels::conv::{
    GATED_DELTA_CONV_OUTPUT_ELEMENT, GatedDeltaConvBlueprint, GatedDeltaConvInputs,
    GatedDeltaConvOutputs, GatedDeltaConvPast, GatedDeltaConvProblem, GatedDeltaConvTensors,
};
use crate::kernels::refusals::admit_on_device;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::gated_delta::GatedDeltaConvPartitioning;
use cubek_microkernel::gated_delta::conv::GatedDeltaConvOperand;
use cubek_microkernel::gated_delta::conv::cube::{
    CubeGatedDeltaConvMicrokernel, GatedDeltaConvSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, PreparedLaunches, Routine};

/// The conv with a key head of a row, or of a position of a sequence, in each cube, its units
/// dealing the head's channels a line at a time. It reads no plane feature, so it is the floor
/// every device runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeGatedDeltaConvRoutine;

impl CubeGatedDeltaConvRoutine {
    /// The conv of `inputs`, launched on their client's device, prepared for `hw`.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the conv on these inputs.
    pub fn launch_on(
        &self,
        hw: &Hardware,
        inputs: GatedDeltaConvInputs,
    ) -> Result<GatedDeltaConvOutputs, Refusal> {
        let tensors = GatedDeltaConvTensors::new(inputs)?;
        let (blueprint, grid) = self.prepare_first_served(hw, tensors.problem())?;
        Ok(self.launch(tensors, blueprint, grid))
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
        inputs: GatedDeltaConvInputs,
    ) -> Result<GatedDeltaConvOutputs, Refusal> {
        let tensors = GatedDeltaConvTensors::new(inputs)?;
        let (blueprint, grid) =
            prepared.prepare(self, hw, tensors.problem(), &GatedDeltaConvSettings)?;
        Ok(self.launch(tensors, blueprint, grid))
    }

    fn launch(
        &self,
        tensors: GatedDeltaConvTensors,
        blueprint: GatedDeltaConvBlueprint,
        grid: Grid,
    ) -> GatedDeltaConvOutputs {
        let binding = tensors.bind(&blueprint, &grid);
        // SAFETY: every operand is bound through the launcher and indexed only through its tile,
        // whose reads past an edge are masked. A unit writes only the window channels it reads, and
        // a cube only its own position and key head of the outputs.
        unsafe {
            cube_gated_delta_conv_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.qkv,
                binding.filters,
                binding.window.into(),
                binding.lengths.into(),
                binding.query,
                binding.key,
                binding.value,
                tensors.problem().past.columns(),
                tensors.epsilon(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                tensors.problem().element,
            );
        }
        tensors.into_outputs()
    }
}

/// Whether `hw` runs `problem`: what a caller asks before it falls back to composed ops.
///
/// # Errors
///
/// [`ELEMENT_NOT_ON_DEVICE`](cubek_matmul::kernels::tensor::ELEMENT_NOT_ON_DEVICE) where the
/// device lacks the activations' element or `f32`, or what [`GatedDeltaConvProblem::admit`]
/// declines.
pub fn serves(hw: &Hardware, problem: &GatedDeltaConvProblem) -> Result<(), Refusal> {
    GatedDeltaConvProblem::admit_element(problem.element)?;
    admit_on_device(hw, [problem.element, GATED_DELTA_CONV_OUTPUT_ELEMENT])?;
    problem.admit(hw)
}

impl Routine for CubeGatedDeltaConvRoutine {
    type Problem = GatedDeltaConvProblem;
    type Microkernel = CubeGatedDeltaConvMicrokernel;
    type Settings = GatedDeltaConvSettings;
    type Blueprint = GatedDeltaConvBlueprint;
    type Partitioning = GatedDeltaConvPartitioning;

    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeGatedDeltaConvMicrokernel::settings(hw)
    }

    fn vectorize(
        &self,
        microkernel: &CubeGatedDeltaConvMicrokernel,
        _: &GatedDeltaConvPartitioning,
        vectorization: &mut Vectorization<GatedDeltaConvOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn estimate(
        &self,
        problem: &GatedDeltaConvProblem,
        blueprint: &GatedDeltaConvBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let heads = &problem.heads;
        let channels = heads.channels_per_key_head();
        let size = problem.element.size();
        let output_size = GATED_DELTA_CONV_OUTPUT_ELEMENT.size();
        let stream = |elements: usize, run: usize, element_bytes: usize| GlobalStream {
            bytes: (elements * element_bytes) as u64,
            run_bytes: run * element_bytes,
        };
        let outputs = stream(channels, heads.key_features, output_size);
        let filters = stream(channels * problem.taps, problem.taps, size);
        let (reads, writes) = match problem.past {
            GatedDeltaConvPast::Window { .. } => (
                vec![
                    stream(channels, heads.key_features, size),
                    filters,
                    stream(channels * (problem.taps - 1), heads.key_features, size),
                ],
                vec![stream(channels, heads.key_features, size), outputs],
            ),
            GatedDeltaConvPast::Stream { .. } => (
                vec![
                    stream(channels * problem.taps, heads.key_features, size),
                    filters,
                ],
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
        problem: &GatedDeltaConvProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        GatedDeltaConvBlueprint::prepare(hw, problem, settings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::conv::GatedDeltaConvPast;
    use crate::kernels::conv::fixtures::{binding_enough, fleet};
    use core::num::NonZeroUsize;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::gated_delta::GatedDeltaHeads;
    use cubek_microkernel::gated_delta::HEAD_CHANNEL;
    use cubek_std::devices;
    use cubek_std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn conv(rows: usize, heads: [usize; 4], taps: usize) -> GatedDeltaConvProblem {
        let [key_heads, group, key_features, value_features] = heads;
        let heads = GatedDeltaHeads {
            key_heads,
            group,
            key_features,
            value_features,
        };
        let past = GatedDeltaConvPast::Window {
            slots: NonZeroUsize::new(rows + 3).unwrap(),
            columns: taps + 1,
        };
        GatedDeltaConvProblem::dense(rows, heads, taps, past, F16)
    }

    fn stream(
        batch: usize,
        positions: usize,
        heads: [usize; 4],
        taps: usize,
    ) -> GatedDeltaConvProblem {
        let [key_heads, group, key_features, value_features] = heads;
        let heads = GatedDeltaHeads {
            key_heads,
            group,
            key_features,
            value_features,
        };
        let past = GatedDeltaConvPast::Stream { positions };
        GatedDeltaConvProblem::dense(batch, heads, taps, past, F16)
    }

    fn problems() -> Vec<GatedDeltaConvProblem> {
        vec![
            conv(1, [16, 3, 128, 128], 4),
            conv(8, [16, 3, 128, 128], 4),
            conv(3, [3, 3, 37, 53], 4),
            conv(2, [2, 2, 12, 8], 2),
            conv(5, [1, 2, 16, 37], 3),
            stream(1, 512, [16, 1, 128, 128], 4),
            stream(1, 512, [16, 3, 128, 128], 4),
            stream(2, 67, [3, 3, 37, 53], 4),
            stream(3, 1, [2, 2, 12, 8], 2),
            stream(1, 9, [1, 2, 16, 37], 3),
        ]
    }

    /// The floor prepares every problem on every device of the fleet, admission is that
    /// preparation, and the launch is the microkernel's: its units no more than the head's lines,
    /// its shared memory, and an estimate that holds the work.
    #[test]
    fn the_floor_prepares_everything_admission_admits() {
        for hw in fleet() {
            for problem in problems() {
                let prepared = CubeGatedDeltaConvRoutine.prepare_first_served(&hw, &problem);
                assert_eq!(problem.admit(&hw), prepared.clone().map(|_| ()));
                let (blueprint, grid) =
                    prepared.unwrap_or_else(|why| panic!("{problem:?}: {why:?}"));
                assert_eq!(grid.shared_memory, blueprint.microkernel.shared_memory());
                assert_eq!(
                    grid.launch.cube_dim.num_elems() as usize,
                    blueprint.microkernel.units
                );
                let lines = problem.heads.channels_per_key_head() / blueprint.width;
                assert_eq!(
                    blueprint.partitioning.as_cubek().distributed(HEAD_CHANNEL),
                    Some(lines)
                );
                assert!(blueprint.microkernel.units <= lines);
                let positions = match problem.past {
                    GatedDeltaConvPast::Window { .. } => 1,
                    GatedDeltaConvPast::Stream { positions } => positions,
                };
                let cubes = problem.batch * positions * problem.heads.key_heads;
                assert_eq!(grid.cubes(), Some(cubes as u64));
                let mut estimate = KernelEstimate::from(&grid);
                CubeGatedDeltaConvRoutine.estimate(&problem, &blueprint, &grid, &mut estimate);
                let staging = estimate.staging.as_ref().expect("staging");
                let performed = staging.operations * estimate.cubes.expect("a static grid") as f64;
                assert!(performed >= problem.demand().compute_ops as f64 * 0.5);
            }
        }
    }

    /// The positions of a stream ride the grid, so admission declines what a device's cube count
    /// cannot hold.
    #[test]
    fn admission_follows_the_grid_a_device_holds() {
        let hw = binding_enough(devices::wgpu_no_subgroups());
        assert_eq!(hw.properties.hardware.max_cube_count.0, 65535);
        assert!(stream(1, 65535, [1, 1, 16, 16], 4).admit(&hw).is_ok());
        assert!(stream(1, 70000, [1, 1, 16, 16], 4).admit(&hw).is_err());
    }
}
