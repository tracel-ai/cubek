use crate::attention::AttentionSlotBox;
use crate::attention::hypercube::HypercubeMergeShape;
use crate::attention::merge::plane::plane_merge_attention_microkernel;
use cubecl::{self as cubecl, cube, prelude::*};

/// The cube that stored its partial of `slot_box`'s queries counting itself in at the box, and the
/// last of the box's cubes to count merging the box into `out`, by one plane
/// ([`plane_merge_attention_microkernel()`])
/// ([`HypercubeMerge::LastArrival`](crate::attention::hypercube::HypercubeMerge::LastArrival)).
///
/// `arrivals` holds a counter for every box, the head's index times the boxes along the queries
/// plus the box's, at zero; the box's last cube leaves it at zero again. The cube meets on
/// `sync_storage` before it counts, and on one `sync_cube`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_attention_arrive<EA: Float, E: Float, N: Size>(
    slot_box: &AttentionSlotBox,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    out: &mut Tensor<E>,
    arrivals: &[Atomic<u32>],
    #[comptime] merge: HypercubeMergeShape,
) {
    let box_queries = comptime!(merge.queries);
    let counter = slot_box.head_index * slot_max.shape(3).div_ceil(box_queries)
        + slot_box.first_query / box_queries;
    // The partial lands at device scope before the cube counts itself in.
    sync_storage();
    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if UNIT_POS == 0 {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }
    sync_cube();
    // Uniform across the cube: only the box's last cube merges.
    if arrived[0] as usize + 1 == comptime!(merge.cubes) {
        // What every other cube of the box stored is acquired before it is read.
        sync_storage();
        plane_merge_attention_microkernel::<EA, E, N>(
            slot_out,
            slot_max,
            slot_sum,
            out,
            slot_box,
            merge.slots,
            merge.queries,
        );
        if UNIT_POS == 0 {
            arrivals[counter].store(0u32);
        }
    }
}
