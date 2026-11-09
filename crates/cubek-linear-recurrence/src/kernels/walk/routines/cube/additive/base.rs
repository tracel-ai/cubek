use super::super::base::{self as shared, CubeWalkRoutineSettings};
use super::kernel::cube_additive_walk_kernel;
use crate::kernels::walk::{
    RecurrenceWalkBinding, RecurrenceWalkBlueprint, RecurrenceWalkProblem, RecurrenceWalkRoutine,
    RecurrenceWalkTensors, UpdateRuleBinding,
};
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::linear_recurrence::RecurrenceWalkPartitioning;
use cubek_microkernel::linear_recurrence::walk::cube::{
    CubeAdditiveWalkMicrokernel, CubeWalkMicrokernel,
};
use cubek_microkernel::linear_recurrence::walk::{RecurrenceWalkOperand, UpdateRule};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// The additive walk placed as the delta walk is: every run of a head's value columns held by a few
/// units, a cube holding several runs of one head side by side, with 9 bindings, 10 in a store's
/// slots, its skip the one more. It declines a problem of the delta rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeAdditiveWalkRoutine;

pub type CubeAdditiveWalkRoutineBlueprint = RecurrenceWalkBlueprint<CubeAdditiveWalkMicrokernel>;

impl Routine for CubeAdditiveWalkRoutine {
    type Problem = RecurrenceWalkProblem;
    type Microkernel = CubeAdditiveWalkMicrokernel;
    type Settings = CubeWalkRoutineSettings;
    type Blueprint = CubeAdditiveWalkRoutineBlueprint;
    type Partitioning = RecurrenceWalkPartitioning;

    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        CubeWalkRoutineSettings::listed()
    }

    /// Nothing of its own: it walks cubes and runs, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeAdditiveWalkMicrokernel,
        _: &RecurrenceWalkPartitioning,
        vectorization: &mut Vectorization<RecurrenceWalkOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn estimate(
        &self,
        problem: &RecurrenceWalkProblem,
        blueprint: &CubeAdditiveWalkRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        shared::estimate(problem, blueprint, grid, estimate);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RecurrenceWalkProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        shared::prepare(hw, problem, settings)
    }
}

impl RecurrenceWalkRoutine<CubeAdditiveWalkMicrokernel> for CubeAdditiveWalkRoutine {
    const NAME: &'static str = "cube-additive-walk";
    const RULE: UpdateRule = CubeAdditiveWalkMicrokernel::RULE;

    fn dispatch(
        &self,
        tensors: &RecurrenceWalkTensors,
        binding: RecurrenceWalkBinding,
        blueprint: CubeAdditiveWalkRoutineBlueprint,
        grid: Grid,
    ) {
        let UpdateRuleBinding::Additive { skip } = binding.update else {
            unreachable!("a problem of the additive rule binds its skip");
        };
        // SAFETY: every operand is bound through the launcher, and the kernel indexes no buffer but
        // through its tiles, which mask what the units overhang. Its shared memory is indexed by
        // its runs' and units' own positions, within the slots it declares.
        unsafe {
            cube_additive_walk_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.query,
                binding.key,
                binding.value,
                binding.rates,
                skip,
                binding.state,
                binding.out,
                binding.kept.into(),
                binding.lengths.into(),
                tensors.problem().first_kept(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.runs,
                tensors.problem().output_element,
            );
        }
    }
}
