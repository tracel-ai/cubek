use super::kernel::{
    hypercube_attention_arrival_kernel, hypercube_attention_arrival_packed_kernel,
};
use crate::kernels::{
    AttentionCacheBinding, AttentionRoutine, AttentionTensors, HypercubeMerge,
    HypercubeShuffleAttentionRoutine, HypercubeShuffleAttentionRoutineBlueprint,
    HypercubeShuffleMerge, HypercubeShuffleOperands, PackedCacheBinding,
};
use cubecl::tile::launch::Decline;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_microkernel::attention::hypercube::HypercubeAttentionMicrokernel;
use cubek_std::routine::Grid;

/// The hypercube shuffle merged in its own launch: every cube counts itself in at its box, and
/// the last to arrive merges the box's slots into the output ([`HypercubeMerge::LastArrival`]).
/// One launch, where a partial routine and its merge take two, but one plane merges a whole box.
pub type HypercubeLastArrivalShuffleAttentionRoutine =
    HypercubeShuffleAttentionRoutine<HypercubeLastArrivalShuffle>;

/// The merge of [`HypercubeLastArrivalShuffleAttentionRoutine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalShuffle;

impl HypercubeShuffleMerge for HypercubeLastArrivalShuffle {
    const MERGE: HypercubeMerge = HypercubeMerge::LastArrival;
    const NAME: &'static str = "hypercube-last-arrival-shuffle-attention";
}

impl AttentionRoutine<HypercubeAttentionMicrokernel, StageStorage>
    for HypercubeLastArrivalShuffleAttentionRoutine
{
    /// Every cube leaving its partial in its slot and counting itself in at its box, the last of a
    /// box's cubes merging the box. The keys and values are tiles, or a packed cache decoded as
    /// the planes read it.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
    ) -> Result<(), Decline> {
        let binding = tensors.bind(&blueprint, &grid)?;
        let (operands, cache) = HypercubeShuffleOperands::of(binding, grid);
        match cache {
            AttentionCacheBinding::Tiles { key, value } => {
                self.arrive_over_tiles(tensors, operands, key, value, &blueprint)
            }
            AttentionCacheBinding::Packed(cache) => {
                self.arrive_over_packed(tensors, operands, cache, &blueprint)
            }
        }
        Ok(())
    }
}

impl HypercubeLastArrivalShuffleAttentionRoutine {
    /// The launch over keys and values bound as tiles.
    fn arrive_over_tiles(
        &self,
        tensors: &AttentionTensors,
        operands: HypercubeShuffleOperands,
        key: TileTensorLaunch,
        value: TileTensorLaunch,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
    ) {
        let problem = tensors.problem();
        let shape = self.merge_shape(&problem, blueprint);
        let microkernel = blueprint.microkernel;
        let slots = tensors.split_slots(shape.slots.splits);
        let arrivals = tensors.box_arrivals(microkernel.cube.seq_q());
        let HypercubeShuffleOperands {
            launcher,
            query,
            mask,
            lengths,
            grid,
        } = operands;
        // SAFETY: the query, keys and values are bound through the launcher, which masks the
        // reads of a tile the problem's edge cuts short, and the keys and values are read no
        // further than the attended length, which is capped at the keys there are. A cube writes
        // no slot but its own, for the queries of its box the problem holds; the last cube of a
        // box writes the box's queries of the output, at its own strides, and reads no slot but
        // its box's, acquired after every cube of the box stored its own where the counters start
        // at zero, as every launch leaves them. A box whose counter does not is merged early or
        // never, which costs its answer: no access leaves its buffer.
        unsafe {
            hypercube_attention_arrival_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.widths.query,
                query,
                key,
                value,
                tensors.out_arg(),
                tensors.logsumexp_arg().into(),
                slots.out.binding().into_tensor_arg(),
                slots.max.binding().into_tensor_arg(),
                slots.sum.binding().into_tensor_arg(),
                arrivals.binding.into_tensor_arg().into_buffer_arg(),
                mask.map(|mask| mask.arg()).into(),
                lengths,
                problem.seq_q as u32,
                problem.seq_kv as u32,
                tensors.scale(),
                launcher.partitioning_arg(),
                microkernel,
                shape,
                problem.visibility,
                problem.work_item().operand_elements(),
            );
        }
    }

    /// The launch over a packed cache, which the planes decode as they read it.
    fn arrive_over_packed(
        &self,
        tensors: &AttentionTensors,
        operands: HypercubeShuffleOperands,
        cache: PackedCacheBinding,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
    ) {
        let problem = tensors.problem();
        let shape = self.merge_shape(&problem, blueprint);
        let microkernel = blueprint.microkernel;
        let slots = tensors.split_slots(shape.slots.splits);
        let arrivals = tensors.box_arrivals(microkernel.cube.seq_q());
        let HypercubeShuffleOperands {
            launcher,
            query,
            mask,
            lengths,
            grid,
        } = operands;
        // SAFETY: as the tiles' launch; a field indexes its codebook, which holds an entry for
        // every index (`PackedAttentionCache::admit`).
        unsafe {
            hypercube_attention_arrival_packed_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.widths.query,
                query,
                cache.key,
                cache.value,
                cache.fixup.into(),
                tensors.out_arg(),
                slots.out.binding().into_tensor_arg(),
                slots.max.binding().into_tensor_arg(),
                slots.sum.binding().into_tensor_arg(),
                arrivals.binding.into_tensor_arg().into_buffer_arg(),
                mask.map(|mask| mask.arg()).into(),
                lengths,
                problem.seq_q as u32,
                problem.seq_kv as u32,
                tensors.scale(),
                launcher.partitioning_arg(),
                microkernel,
                shape,
                problem.visibility,
                problem.work_item().kernel_elements(),
            );
        }
    }
}
