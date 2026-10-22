use cubecl::{self as cubecl, cube, prelude::*};

/// Partials of the same rows left in slots, each by a worker that walked its own part of the keys:
/// a row's slots lie side by side, `splits` of them, each a running max, a running sum, and an
/// output of `value` elements, unnormalized, in lines of the operands' width. A row's slot `t`
/// lies at `row · splits + t` of the maxima and the sums, and its output's line `l` at that slot
/// times the lines of a value, plus `l`.
///
/// What a merge of them is compiled for, beside the units of the plane that merges a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionSlotShape {
    /// Units of the plane that merges a row, which share its work.
    pub units: usize,
    /// Elements of a row's output.
    pub val_dim: usize,
    /// Slots of every row.
    pub splits: usize,
}

/// A box of queries of one head as the slots hold them: the first query's position as the
/// output's `(b, kv, g, q)`, its index in the slots' order, and how many of the box's queries the
/// problem holds.
///
/// Query `q` of head `g` of key head `kv` in batch `b` is one row of the slots, and each of the
/// `splits` cubes sharing its box holds a slot of it, `t`: the slots' max and sum lie at
/// `(((b · kv_heads + kv) · group + g) · queries + q) · splits + t`, shaped `[b, kv, g, q,
/// splits]`, and the slots' output holds the row's lines there as
/// [`AttentionSlotShape`](crate::attention::AttentionSlotShape) lays them out.
#[derive(CubeType, Clone)]
pub struct AttentionSlotBox {
    pub batch: usize,
    pub kv: usize,
    pub head: usize,
    pub first_query: usize,
    /// The head's index in the slots' order, `(b · kv_heads + kv) · group + g`.
    pub head_index: usize,
    /// The first query's row of the slots.
    pub first_row: usize,
    /// Queries of the box the problem holds, from the first: none for a head past the group,
    /// which a cube holding several heads overhangs where they do not divide it.
    pub seq_q: usize,
}

#[cube]
impl AttentionSlotBox {
    /// The box of `queries` queries of head `head` of key head `kv` in batch `batch` from query
    /// position `first_query`, in slots shaped as `slot_max`.
    pub fn new<EA: Float>(
        // Read only, taken as the kernel holds it: cube code passes no `&mut` where a `&` is asked.
        slot_max: &mut Tensor<EA>,
        batch: usize,
        kv: usize,
        head: usize,
        first_query: usize,
        #[comptime] queries: usize,
    ) -> AttentionSlotBox {
        let (kv_heads, group, held_queries) =
            (slot_max.shape(1), slot_max.shape(2), slot_max.shape(3));
        let mut held = 0usize;
        if head < group && held_queries > first_query {
            held = held_queries - first_query;
        }
        let head_index = (batch * kv_heads + kv) * group + head;
        AttentionSlotBox {
            batch,
            kv,
            head,
            first_query,
            head_index,
            first_row: head_index * held_queries + first_query,
            seq_q: held.min(queries),
        }
    }
}
