use super::kernel::{
    hypercube_attention_arrival_kernel, hypercube_attention_arrival_packed_kernel,
};
use crate::kernels::{
    AttentionCacheBinding, AttentionRoutine, AttentionTensors, HypercubeMerge,
    HypercubeShuffleAttentionRoutine, HypercubeShuffleAttentionRoutineBlueprint,
    HypercubeShuffleLaunch, HypercubeShuffleMerge, PackedCacheBinding,
};
use cubecl::tile::launch::Decline;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_matmul::kernels::tensor::PlainTensor;
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
    /// the planes read it. The slots and the counters are allocated before the launch.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
    ) -> Result<(), Decline> {
        let binding = tensors.bind(&blueprint, &grid)?;
        let shape = self.merge_shape(&tensors.problem(), &blueprint);
        let slots = tensors.split_slots(shape.slots.splits);
        let arrivals = tensors.box_arrivals(blueprint.microkernel.cube.seq_q());
        let (launch, cache) =
            HypercubeShuffleLaunch::new(tensors, binding, &blueprint, grid, shape, &slots);
        match cache {
            AttentionCacheBinding::Tiles { key, value } => {
                arrive_over_tiles(tensors, launch, key, value, arrivals)
            }
            AttentionCacheBinding::Packed(cache) => {
                arrive_over_packed(tensors, launch, cache, arrivals)
            }
        }
        Ok(())
    }
}

/// `launch` over keys and values bound as tiles, its cubes counting themselves in at `arrivals`.
fn arrive_over_tiles(
    tensors: &AttentionTensors,
    launch: HypercubeShuffleLaunch,
    key: TileTensorLaunch,
    value: TileTensorLaunch,
    arrivals: PlainTensor,
) {
    let [slot_out, slot_max, slot_sum] = launch.slots;
    // SAFETY: the query, keys and values are bound through the launcher, which masks the reads of a
    // tile the problem's edge cuts short, and the keys and values are read no further than the
    // attended length, which is capped at the keys the launch is given, the problem's positions the
    // buffers or their table reach. A cube writes no slot but its own, for the queries of its box
    // the problem holds; the last cube of a box writes the box's queries of the output, at its own
    // strides, and reads no slot but its box's, acquired after every cube of the box stored its own
    // where the counters start at zero, as every launch leaves them. A box whose counter does not
    // is merged early or never, which costs its answer: no access leaves its buffer.
    unsafe {
        hypercube_attention_arrival_kernel::launch_unchecked(
            launch.client,
            launch.cube_count,
            launch.cube_dim,
            launch.width_scalars,
            launch.query,
            key,
            value,
            tensors.out_arg(),
            tensors.logsumexp_arg().into(),
            slot_out,
            slot_max,
            slot_sum,
            arrivals.binding.into_tensor_arg().into_buffer_arg(),
            launch.mask.into(),
            launch.sinks.into(),
            launch.relative.into(),
            launch.lengths,
            launch.query_count,
            launch.key_count,
            launch.scale,
            launch.launcher.partitioning_arg(),
            launch.microkernel,
            launch.shape,
            launch.visibility,
            launch.operand_elements,
        );
    }
}

/// `launch` over a packed cache, which the planes decode as they read it, its cubes counting
/// themselves in at `arrivals`.
fn arrive_over_packed(
    tensors: &AttentionTensors,
    launch: HypercubeShuffleLaunch,
    cache: PackedCacheBinding,
    arrivals: PlainTensor,
) {
    let [slot_out, slot_max, slot_sum] = launch.slots;
    // SAFETY: as the tiles' launch; a field indexes its codebook, which holds an entry for every
    // index (`PackedAttentionCache::admit`).
    unsafe {
        hypercube_attention_arrival_packed_kernel::launch_unchecked(
            launch.client,
            launch.cube_count,
            launch.cube_dim,
            launch.width_scalars,
            launch.query,
            cache.key,
            cache.value,
            cache.fixup.into(),
            tensors.out_arg(),
            slot_out,
            slot_max,
            slot_sum,
            arrivals.binding.into_tensor_arg().into_buffer_arg(),
            launch.mask.into(),
            launch.sinks.into(),
            launch.relative.into(),
            launch.lengths,
            launch.query_count,
            launch.key_count,
            launch.scale,
            launch.launcher.partitioning_arg(),
            launch.microkernel,
            launch.shape,
            launch.visibility,
            launch.kernel_elements,
        );
    }
}
