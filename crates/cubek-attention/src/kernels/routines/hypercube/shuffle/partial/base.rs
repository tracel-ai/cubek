use super::kernel::{
    hypercube_attention_partial_kernel, hypercube_attention_partial_packed_kernel,
};
use crate::kernels::{
    AttentionCacheBinding, AttentionRoutineProblem, AttentionSplitSlots, AttentionTensors,
    HypercubeMerge, HypercubeShuffleAttentionRoutine, HypercubeShuffleAttentionRoutineBlueprint,
    HypercubeShuffleLaunch, HypercubeShuffleMerge, PackedCacheBinding, PartialAttentionRoutine,
};
use cubecl::tile::launch::Decline;
use cubecl::tile::{StageStorage, TileTensorLaunch};
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
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
        slots: &AttentionSplitSlots,
    ) -> Result<(), Decline> {
        let binding = tensors.bind(&blueprint, &grid)?;
        let shape =
            HypercubeShuffleAttentionRoutine::merge_shape(self, &tensors.problem(), &blueprint);
        let (launch, cache) =
            HypercubeShuffleLaunch::new(tensors, binding, &blueprint, grid, shape, slots);
        match cache {
            AttentionCacheBinding::Tiles { key, value } => {
                leave_over_tiles(tensors, launch, key, value)
            }
            AttentionCacheBinding::Packed(cache) => leave_over_packed(tensors, launch, cache),
        }
        Ok(())
    }
}

/// `launch` over keys and values bound as tiles.
fn leave_over_tiles(
    tensors: &AttentionTensors,
    launch: HypercubeShuffleLaunch,
    key: TileTensorLaunch,
    value: TileTensorLaunch,
) {
    let [slot_out, slot_max, slot_sum] = launch.slots;
    // SAFETY: the query, keys and values are bound through the launcher, which masks the reads of
    // a tile the problem's edge cuts short, and the keys and values are read no further than the
    // attended length, which is capped at the keys there are. A cube writes no slot but its own,
    // for the queries of its box the problem holds.
    unsafe {
        hypercube_attention_partial_kernel::launch_unchecked(
            launch.client,
            launch.cube_count,
            launch.cube_dim,
            launch.width,
            launch.query,
            key,
            value,
            slot_out,
            slot_max,
            slot_sum,
            launch.mask.into(),
            launch.lengths,
            launch.queries,
            launch.keys,
            launch.scale,
            launch.launcher.partitioning_arg(),
            launch.microkernel,
            launch.shape,
            launch.visibility,
            tensors.problem().work_item().operand_elements(),
        );
    }
}

/// `launch` over a packed cache, which the planes decode as they read it.
fn leave_over_packed(
    tensors: &AttentionTensors,
    launch: HypercubeShuffleLaunch,
    cache: PackedCacheBinding,
) {
    let [slot_out, slot_max, slot_sum] = launch.slots;
    // SAFETY: as the tiles' launch; a field indexes its codebook, which holds an entry for every
    // index (`PackedAttentionCache::admit`).
    unsafe {
        hypercube_attention_partial_packed_kernel::launch_unchecked(
            launch.client,
            launch.cube_count,
            launch.cube_dim,
            launch.width,
            launch.query,
            cache.key,
            cache.value,
            cache.fixup.into(),
            slot_out,
            slot_max,
            slot_sum,
            launch.mask.into(),
            launch.lengths,
            launch.queries,
            launch.keys,
            launch.scale,
            launch.launcher.partitioning_arg(),
            launch.microkernel,
            launch.shape,
            launch.visibility,
            tensors.problem().work_item().kernel_elements(),
        );
    }
}
