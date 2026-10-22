use super::kernel::{
    hypercube_attention_partial_kernel, hypercube_attention_partial_packed_kernel,
};
use crate::kernels::{
    AttentionBinding, AttentionCacheBinding, AttentionRoutineProblem, AttentionTensors,
    HypercubeSecondLaunch, HypercubeShuffleAttentionRoutine,
    HypercubeShuffleAttentionRoutineBlueprint, HypercubeShuffleMerge,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::hypercube::{HypercubeMerge, HypercubeMergeShape};
use cubek_std::Hardware;
use cubek_std::routine::Grid;

/// The hypercube shuffle merged by a second launch, a plane a query
/// ([`HypercubeMerge::SecondLaunch`]): every query merged at once rather than one plane a box,
/// and no cube reading another's stores, so it runs on every device that has a plane width.
pub type HypercubeSecondLaunchShuffleAttentionRoutine =
    HypercubeShuffleAttentionRoutine<HypercubeSecondLaunchShuffle>;

/// The merge of [`HypercubeSecondLaunchShuffleAttentionRoutine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeSecondLaunchShuffle;

impl HypercubeShuffleMerge for HypercubeSecondLaunchShuffle {
    const MERGE: HypercubeMerge = HypercubeMerge::SecondLaunch;

    /// The second launch, checked against the device.
    fn validate(
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        shape: HypercubeMergeShape,
    ) -> Result<(), Refusal> {
        HypercubeSecondLaunch::new(problem, shape).validate(hw)
    }

    /// Two launches: every cube leaving its partial in its slot, then a plane for every query
    /// merging its slots into the output.
    fn dispatch(
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
        shape: HypercubeMergeShape,
    ) {
        let problem = tensors.problem();
        let microkernel = blueprint.microkernel;
        let slots = tensors.split_slots(shape.slots.splits);
        let width = blueprint.widths.query;
        let (client, count, dim) = (
            tensors.client(),
            grid.launch.cube_count,
            grid.launch.cube_dim,
        );
        let partitioning = binding.launcher.partitioning_arg();
        let (queries, scale) = (problem.queries as u32, tensors.scale());
        let elements = problem.work_item().kernel_elements();
        match binding.cache {
            AttentionCacheBinding::Tiles { key, value } => {
                // SAFETY: the query, keys and values are bound through the launcher, which masks
                // the reads of a tile the problem's edge cuts short, and the keys and values are
                // read no further than the attended length. A cube writes no slot but its own, for
                // the queries of its box the problem holds.
                unsafe {
                    hypercube_attention_partial_kernel::launch_unchecked(
                        client,
                        count,
                        dim,
                        width,
                        binding.query.arg(),
                        key.arg(),
                        value.arg(),
                        slots.out.clone().binding().into_tensor_arg(),
                        slots.max.clone().binding().into_tensor_arg(),
                        slots.sum.clone().binding().into_tensor_arg(),
                        binding.lengths,
                        binding.pages,
                        queries,
                        scale,
                        partitioning,
                        microkernel,
                        shape,
                        problem.visibility,
                        elements,
                    );
                }
                HypercubeSecondLaunch::new(&problem, shape).launch(tensors, slots, width);
            }
            AttentionCacheBinding::Packed(cache) => {
                // SAFETY: as the tiles'; the packed words, norms and codebooks are read no further
                // than the attended keys, and a field indexes its codebook, whose size is its
                // width's.
                unsafe {
                    hypercube_attention_partial_packed_kernel::launch_unchecked(
                        client,
                        count,
                        dim,
                        width,
                        binding.query.arg(),
                        cache,
                        slots.out.clone().binding().into_tensor_arg(),
                        slots.max.clone().binding().into_tensor_arg(),
                        slots.sum.clone().binding().into_tensor_arg(),
                        binding.lengths,
                        queries,
                        scale,
                        partitioning,
                        microkernel,
                        shape,
                        problem.visibility,
                        elements,
                    );
                }
                HypercubeSecondLaunch::new(&problem, shape).launch(tensors, slots, width);
            }
        }
    }
}
