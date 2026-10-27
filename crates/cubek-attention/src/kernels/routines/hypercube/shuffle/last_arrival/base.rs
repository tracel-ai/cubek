use super::kernel::{
    hypercube_attention_arrival_kernel, hypercube_attention_arrival_packed_kernel,
};
use crate::kernels::{
    AttentionBinding, AttentionCacheBinding, AttentionRoutineProblem, AttentionTensors,
    HypercubeShuffleAttentionRoutine, HypercubeShuffleAttentionRoutineBlueprint,
    HypercubeShuffleMerge,
};
use cubecl::prelude::ComptimeOptionArgs;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::hypercube::{HypercubeMerge, HypercubeMergeShape};
use cubek_std::Hardware;
use cubek_std::routine::Grid;

/// The hypercube shuffle merged in its own launch: every cube counts itself in at its box, and
/// the last to arrive merges the box's slots into the output ([`HypercubeMerge::LastArrival`]).
/// One launch, where a second launch takes two, but one plane merges a whole box.
pub type HypercubeLastArrivalShuffleAttentionRoutine =
    HypercubeShuffleAttentionRoutine<HypercubeLastArrivalShuffle>;

/// The merge of [`HypercubeLastArrivalShuffleAttentionRoutine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalShuffle;

impl HypercubeShuffleMerge for HypercubeLastArrivalShuffle {
    const MERGE: HypercubeMerge = HypercubeMerge::LastArrival;

    /// Nothing beyond the device: the merge is the cubes' own.
    fn validate(
        _hw: &Hardware,
        _problem: &AttentionRoutineProblem,
        _shape: HypercubeMergeShape,
    ) -> Result<(), Refusal> {
        Ok(())
    }

    /// One launch, every cube leaving its partial in its slot and counting itself in at its box,
    /// the last of a box's cubes merging the box.
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
        let (queries, scale) = (problem.seq_q as u32, tensors.scale());
        let elements = problem.work_item().kernel_elements();
        let arrivals = tensors.box_arrivals(microkernel.cube.seq_q());
        match binding.cache {
            AttentionCacheBinding::Tiles { key, value } => {
                // SAFETY: the query, keys and values are bound through the launcher, which masks
                // the reads of a tile the problem's edge cuts short, and the keys and values are
                // read no further than the attended length. A cube writes no slot but its own, for
                // the queries of its box the problem holds; the last cube of a box writes the
                // box's queries of the output, at its own strides, and reads no slot but its
                // box's, acquired after every cube of the box stored its own. The counters hold
                // zero, and every launch leaves them so.
                unsafe {
                    hypercube_attention_arrival_kernel::launch_unchecked(
                        client,
                        count,
                        dim,
                        width,
                        binding.query.arg(),
                        key,
                        value,
                        tensors.out_arg(),
                        slots.out.binding().into_tensor_arg(),
                        slots.max.binding().into_tensor_arg(),
                        slots.sum.binding().into_tensor_arg(),
                        arrivals.binding.into_tensor_arg().into_buffer_arg(),
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
            }
            AttentionCacheBinding::Packed(cache) => {
                let fixup = match cache.fixup {
                    Some(fixup) => ComptimeOptionArgs::Some(fixup),
                    None => ComptimeOptionArgs::None,
                };
                // SAFETY: as the tiles'; a field indexes its codebook, which holds an entry for
                // every index (`PackedAttentionCache::new`).
                unsafe {
                    hypercube_attention_arrival_packed_kernel::launch_unchecked(
                        client,
                        count,
                        dim,
                        width,
                        binding.query.arg(),
                        cache.key,
                        cache.value,
                        fixup,
                        tensors.out_arg(),
                        slots.out.binding().into_tensor_arg(),
                        slots.max.binding().into_tensor_arg(),
                        slots.sum.binding().into_tensor_arg(),
                        arrivals.binding.into_tensor_arg().into_buffer_arg(),
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
            }
        }
    }
}
