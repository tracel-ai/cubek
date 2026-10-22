use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::AttentionSlotBox;
use cubek_microkernel::attention::hypercube::HypercubeMergeShape;
use cubek_microkernel::attention::merge::plane::plane_merge_attention_microkernel;

/// A plane for every query merging the query's slots into the output: the query is `q = x`, head
/// `g = y % group` of key head `kv = y / group`, batch `b = z` of the cube's position.
#[cube(launch_unchecked)]
pub fn hypercube_attention_merge_kernel<EA: Float, E: Float, N: Size>(
    out: &mut Tensor<E>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    #[comptime] merge: HypercubeMergeShape,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let group = slot_max.shape(2);
    let (batch, query_head, query) = (
        CUBE_POS_Z as usize,
        CUBE_POS_Y as usize,
        CUBE_POS_X as usize,
    );
    let slot_box = AttentionSlotBox::new::<EA>(
        slot_max,
        batch,
        query_head / group,
        query_head % group,
        query,
        comptime!(merge.queries),
    );
    plane_merge_attention_microkernel::<EA, E, N>(
        slot_out,
        slot_max,
        slot_sum,
        out,
        &slot_box,
        merge.slots,
        merge.queries,
    );
}
