use super::kernel::plane_split_keys_merge_kernel;
use crate::kernels::{AttentionRoutineProblem, AttentionSplitSlots, AttentionTensors};
use cubecl::tile::launch::Refusal;
use cubecl::{self as cubecl, prelude::*};
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_std::routine::LaunchSettings;

/// The split-keys merge: a cube of one plane for every query, at `(q, kv · group + g, b)`, merging
/// the slots a partial attention routine left for the query
/// ([`PartialAttentionRoutine`](crate::kernels::PartialAttentionRoutine)) into the output, and
/// into its logsumexp where the tensors ask for it.
///
/// Every query is merged at once, and no cube waits on another. What a strategy launches after the
/// partial routine ([`SplitKeysAttention`](crate::kernels::SplitKeysAttention)).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneSplitKeysMergeRoutine;

/// What the merge compiles and launches: how the slots lie, the width their outputs' lines are
/// stored at, and its grid.
#[derive(Debug, Clone)]
pub struct PlaneSplitKeysMergeBlueprint {
    pub merge: HypercubeMergeShape,
    pub slot_width: usize,
    pub launch: LaunchSettings,
}

impl PlaneSplitKeysMergeRoutine {
    /// The merge of `problem`'s slots, laid out as `merge` says, their outputs' lines
    /// `slot_width` elements wide.
    ///
    /// # Errors
    ///
    /// A cube or a grid the device cannot spawn.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        merge: HypercubeMergeShape,
        slot_width: usize,
    ) -> Result<PlaneSplitKeysMergeBlueprint, Refusal> {
        let launch = LaunchSettings {
            cube_dim: CubeDim::new_1d(merge.slots.units as u32),
            cube_count: CubeCount::new_3d(
                problem.seq_q as u32,
                (problem.num_kv_heads * problem.group) as u32,
                problem.batch as u32,
            ),
            vectors: (),
        }
        .validate(hw)?;
        Ok(PlaneSplitKeysMergeBlueprint {
            merge,
            slot_width,
            launch,
        })
    }

    /// Every query's `slots` merged into `tensors`' output, and into its logsumexp where the
    /// tensors ask for it.
    pub fn dispatch(
        &self,
        tensors: &AttentionTensors,
        blueprint: PlaneSplitKeysMergeBlueprint,
        slots: AttentionSplitSlots,
    ) {
        // SAFETY: a plane for every query, `(q, kv · group + g, b)`, reads that query's slots, all
        // written by the launch before, and writes that query of the output at its own strides.
        unsafe {
            plane_split_keys_merge_kernel::launch_unchecked(
                tensors.client(),
                blueprint.launch.cube_count,
                blueprint.launch.cube_dim,
                blueprint.slot_width,
                tensors.out_arg(),
                tensors.logsumexp_arg().into(),
                slots.out.binding().into_tensor_arg(),
                slots.max.binding().into_tensor_arg(),
                slots.sum.binding().into_tensor_arg(),
                blueprint.merge,
                tensors.problem().work_item().kernel_elements(),
            );
        }
    }
}
