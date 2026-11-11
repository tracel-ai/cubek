use super::kernel::{
    hypercube_attention_partial_kernel, hypercube_attention_partial_packed_kernel,
};
use crate::kernels::{
    AttentionBinding, AttentionCacheBinding, AttentionRoutineProblem, AttentionSplitSlots,
    AttentionTensors, HypercubeMerge, HypercubeShuffleAttentionRoutine,
    HypercubeShuffleAttentionRoutineBlueprint, HypercubeShuffleMerge, PartialAttentionRoutine,
};
use cubecl::prelude::ComptimeOptionArgs;
use cubecl::tile::StageStorage;
use cubek_microkernel::attention::hypercube::HypercubeAttentionMicrokernel;
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::routine::Grid;

/// The hypercube shuffle stopped at its partials: every cube leaves its run's partial in its slot,
/// and the split-keys merge launched after it makes the output, a plane a query
/// ([`HypercubeMerge::SecondLaunch`]). Every query is merged at once rather than one plane a box,
/// and no cube reads another's stores, so it runs on every device that has a plane width.
pub type HypercubeShufflePartialAttentionRoutine =
    HypercubeShuffleAttentionRoutine<HypercubePartialShuffle>;

/// The merge of [`HypercubeShufflePartialAttentionRoutine`]: none of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubePartialShuffle;

impl HypercubeShuffleMerge for HypercubePartialShuffle {
    const MERGE: HypercubeMerge = HypercubeMerge::SecondLaunch;
    const NAME: &'static str = "hypercube-shuffle-partial-attention";
}

impl PartialAttentionRoutine<HypercubeAttentionMicrokernel, StageStorage>
    for HypercubeShufflePartialAttentionRoutine
{
    fn merge_shape(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
    ) -> HypercubeMergeShape {
        HypercubeShuffleAttentionRoutine::merge_shape(self, problem, blueprint)
    }

    /// The query's: the plane shuffle lines every operand alike.
    fn slot_width(&self, blueprint: &HypercubeShuffleAttentionRoutineBlueprint) -> usize {
        blueprint.widths.query
    }

    /// Every cube leaving its partial in its slot. The keys and values are tiles, or a packed
    /// cache decoded as the planes read it.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
        slots: &AttentionSplitSlots,
    ) {
        Self::assert_one_width(&binding, &blueprint);
        let problem = tensors.problem();
        let shape = self.merge_shape(&problem, &blueprint);
        let microkernel = blueprint.microkernel;
        let width = blueprint.widths.query;
        let (client, count, dim) = (
            tensors.client(),
            grid.launch.cube_count,
            grid.launch.cube_dim,
        );
        let partitioning = binding.launcher.partitioning_arg();
        let (queries, scale) = (problem.seq_q as u32, tensors.scale());
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
                        binding.query,
                        key,
                        value,
                        slots.out.clone().binding().into_tensor_arg(),
                        slots.max.clone().binding().into_tensor_arg(),
                        slots.sum.clone().binding().into_tensor_arg(),
                        binding.mask.map(|mask| mask.arg()).into(),
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
                    hypercube_attention_partial_packed_kernel::launch_unchecked(
                        client,
                        count,
                        dim,
                        width,
                        binding.query,
                        cache.key,
                        cache.value,
                        fixup,
                        slots.out.clone().binding().into_tensor_arg(),
                        slots.max.clone().binding().into_tensor_arg(),
                        slots.sum.clone().binding().into_tensor_arg(),
                        binding.mask.map(|mask| mask.arg()).into(),
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
