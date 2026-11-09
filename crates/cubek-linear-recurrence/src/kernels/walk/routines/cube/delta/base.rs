use super::super::CubeDeltaWalkRoutineBlueprint;
use super::kernel::cube_delta_walk_kernel;
use crate::kernels::walk::{RecurrenceWalkBinding, RecurrenceWalkRoutine, RecurrenceWalkTensors};
use cubek_microkernel::linear_recurrence::walk::cube::CubeDeltaWalkMicrokernel;
use cubek_std::routine::Grid;

/// The walk with every run of a head's value columns held by a few units, a cube holding several
/// runs of one head side by side. It reads no plane feature; it needs units, shared memory and 8
/// bindings, 9 in a store's slots: this kind's floor.
///
/// A unit owns a tile of the run's key rows: the partitioning's leaf holds `rows` of them and its
/// units level cuts `KEY_FEATURE` into one leaf for each of a run's `units` units, a power of two.
/// A unit reads its position from the region it is handed, and every tile it reads or writes is
/// windowed to its rows, so rows the head leaves past its last are masked. A head whose key rows
/// no such units hold within [`MAX_HELD_LINES_PER_UNIT`] lines each is declined.
///
/// Every run of a cube reads the same queries and keys, those of its head's key head, a position
/// at a time, so the cube's runs share those reads through the device's caches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeDeltaWalkRoutine;

impl RecurrenceWalkRoutine<CubeDeltaWalkMicrokernel> for CubeDeltaWalkRoutine {
    fn dispatch(
        &self,
        tensors: &RecurrenceWalkTensors,
        binding: RecurrenceWalkBinding,
        blueprint: CubeDeltaWalkRoutineBlueprint,
        grid: Grid,
    ) {
        // SAFETY: every operand is bound through the launcher, and the kernel indexes no buffer but
        // through its tiles, which mask what the units overhang. Its shared memory is indexed by
        // its runs' and units' own positions, within the slots it declares.
        unsafe {
            cube_delta_walk_kernel::launch_unchecked(
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
