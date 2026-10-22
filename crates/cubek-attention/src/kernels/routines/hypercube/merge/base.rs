use super::kernel::hypercube_attention_merge_kernel;
use crate::kernels::{AttentionRoutineProblem, AttentionSplitSlots, AttentionTensors};
use cubecl::tile::launch::Refusal;
use cubecl::{self as cubecl, prelude::*};
use cubek_microkernel::attention::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_std::routine::LaunchSettings;

/// The second launch of a hypercube routine that merges by one
/// ([`HypercubeMerge::SecondLaunch`](cubek_microkernel::attention::hypercube::HypercubeMerge)):
/// a cube of one plane for every query, at `(q, kv · group + g, b)`, merging the query's slots,
/// all written by the launch before, into the output.
pub struct HypercubeSecondLaunch {
    merge: HypercubeMergeShape,
    launch: LaunchSettings,
}

impl HypercubeSecondLaunch {
    /// The merge of `problem`'s slots, as `merge` shapes them.
    pub fn new(problem: &AttentionRoutineProblem, merge: HypercubeMergeShape) -> Self {
        Self {
            merge,
            launch: LaunchSettings {
                cube_dim: CubeDim::new_1d(merge.slots.units as u32),
                cube_count: CubeCount::new_3d(
                    problem.seq_q as u32,
                    (problem.num_kv_heads * problem.group) as u32,
                    problem.batch as u32,
                ),
                vectors: (),
            },
        }
    }

    /// Whether `hw` runs this launch.
    ///
    /// # Errors
    ///
    /// Why it does not.
    pub fn validate(&self, hw: &Hardware) -> Result<(), Refusal> {
        self.launch.clone().validate(hw).map(|_| ())
    }

    /// Every query's `slots`, their lines `slot_width` elements wide, merged into `tensors`'
    /// output.
    pub fn launch(self, tensors: &AttentionTensors, slots: AttentionSplitSlots, slot_width: usize) {
        // SAFETY: a plane for every query, `(q, kv · group + g, b)`, reads that query's slots, all
        // written by the launch before, and writes that query of the output at its own strides.
        unsafe {
            hypercube_attention_merge_kernel::launch_unchecked(
                tensors.client(),
                self.launch.cube_count,
                self.launch.cube_dim,
                slot_width,
                tensors.out_arg(),
                slots.out.binding().into_tensor_arg(),
                slots.max.binding().into_tensor_arg(),
                slots.sum.binding().into_tensor_arg(),
                self.merge,
                tensors.problem().work_item().kernel_elements(),
            );
        }
    }
}
