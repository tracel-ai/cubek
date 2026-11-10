use super::base::{self as shared, CubeWalkRoutineSettings};
use super::kernel::cube_walk_kernel;
use crate::kernels::walk::{
    RecurrenceWalkBinding, RecurrenceWalkBlueprint, RecurrenceWalkProblem, RecurrenceWalkRoutine,
    RecurrenceWalkTensors, UpdateRuleBinding,
};
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::linear_recurrence::RecurrenceWalkPartitioning;
use cubek_microkernel::linear_recurrence::walk::cube::CubeWalkMicrokernel;
use cubek_microkernel::linear_recurrence::walk::{RecurrenceWalkOperand, UpdateRule};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// The walk of one update rule with every run of a head's value columns held by a few units, a
/// cube holding several runs of one head side by side. It reads no plane feature; it needs units,
/// shared memory and the bindings of its rule: 8, 9 in a store's slots, for the delta rule, one
/// more for the additive rule's skip. It declines a problem of the other rule.
///
/// A unit owns a tile of the run's key rows: the partitioning's leaf holds `rows` of them and its
/// units level cuts `KEY_FEATURE` into one leaf for each of a run's `units` units, a power of two.
/// A unit reads its position from the region it is handed, and every tile it reads or writes is
/// windowed to its rows, so rows the head leaves past its last are masked. A head whose key rows
/// no such units hold within `MAX_HELD_LINES_PER_UNIT` lines each is declined.
///
/// Every run of a cube reads the same queries and keys, those of its head's key head, a position
/// at a time, so the cube's runs share those reads through the device's caches. Both rules place
/// their cubes alike, so the rule is a field: a rule that needs another placement is its own
/// routine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeWalkRoutine {
    rule: UpdateRule,
}

pub type CubeWalkRoutineBlueprint = RecurrenceWalkBlueprint<CubeWalkMicrokernel>;

impl CubeWalkRoutine {
    pub const DELTA: Self = Self::new(UpdateRule::Delta);
    pub const ADDITIVE: Self = Self::new(UpdateRule::Additive);

    /// The routine walking `rule`.
    pub const fn new(rule: UpdateRule) -> Self {
        Self { rule }
    }
}

impl Routine for CubeWalkRoutine {
    type Problem = RecurrenceWalkProblem;
    type Microkernel = CubeWalkMicrokernel;
    type Settings = CubeWalkRoutineSettings;
    type Blueprint = CubeWalkRoutineBlueprint;
    type Partitioning = RecurrenceWalkPartitioning;

    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        CubeWalkRoutineSettings::listed()
    }

    /// Nothing of its own: it walks cubes and runs, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeWalkMicrokernel,
        _: &RecurrenceWalkPartitioning,
        vectorization: &mut Vectorization<RecurrenceWalkOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn estimate(
        &self,
        problem: &RecurrenceWalkProblem,
        blueprint: &CubeWalkRoutineBlueprint,
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
        shared::prepare(self.rule, hw, problem, settings)
    }
}

impl RecurrenceWalkRoutine<CubeWalkMicrokernel> for CubeWalkRoutine {
    fn name(&self) -> &'static str {
        match self.rule {
            UpdateRule::Delta => "cube-delta-walk",
            UpdateRule::Additive => "cube-additive-walk",
        }
    }

    fn rule(&self) -> UpdateRule {
        self.rule
    }

    fn dispatch(
        &self,
        tensors: &RecurrenceWalkTensors,
        binding: RecurrenceWalkBinding,
        blueprint: CubeWalkRoutineBlueprint,
        grid: Grid,
    ) {
        let skip = match binding.update {
            UpdateRuleBinding::Delta => None,
            UpdateRuleBinding::Additive { skip } => Some(skip),
        };
        // SAFETY: every operand is bound through the launcher, and the kernel indexes no buffer but
        // through its tiles, which mask what the units overhang. Its shared memory is indexed by
        // its runs' and units' own positions, within the slots it declares.
        unsafe {
            cube_walk_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.query,
                binding.key,
                binding.value,
                binding.rates,
                binding.state,
                binding.out,
                skip.into(),
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
