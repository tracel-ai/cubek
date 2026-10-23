use crate::attention::{AttentionVisibility, G, Q, S};
use cubecl::tile::{
    Axis, Partitioning, Region, Tile,
    procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
};
use cubecl::{self as cubecl, cube, ir::Scope, prelude::*};

/// Which keys a query sees, and the bias its scores are summed with: zero at a key it sees, the
/// minimum value elsewhere. A microkernel that holds a tile of scores sums the bias into it; one
/// that scores keys in its own registers asks [`sees`](VisibleKeys::sees). A query sees the first
/// `attended` keys; under [`Causal`](AttentionVisibility::Causal) none past its own position: query `q` of
/// `queries` sees key `s` where `s + queries ≤ q + attended`; under
/// [`Windowed`](AttentionVisibility::Windowed) none `window` or more before it either, where
/// `s + queries + window > q + attended`; and under [`Recent`](AttentionVisibility::Recent) the
/// last `window` of them, where `s + window ≥ attended`.
#[derive(CubeType, Clone)]
pub struct VisibleKeys {
    /// Keys attended, a prefix of the buffers': none past it is read.
    pub attended: u32,
    pub seq_q: u32,
    #[cube(comptime)]
    pub visibility: AttentionVisibility,
}

#[cube]
impl VisibleKeys {
    /// What sequence `batch`'s `queries` see: its entry of `len`, read by stride, so a single
    /// length bound at stride zero is every sequence's.
    pub fn new(
        len: &Tensor<u32>,
        batch: usize,
        queries: u32,
        #[comptime] visibility: AttentionVisibility,
    ) -> VisibleKeys {
        VisibleKeys {
            attended: len[batch * len.stride(0)],
            seq_q: queries,
            visibility,
        }
    }

    /// The bias the scores are summed with, over `partitioning`'s heads, queries and keys.
    pub fn bias<E: Numeric>(&self, partitioning: &Partitioning) -> Tile {
        Procedural::new::<E, VisibleKeys>(
            comptime!(partitioning.space().subspace(&[G, Q, S])),
            VisibleKeys {
                attended: self.attended,
                seq_q: self.seq_q,
                visibility: comptime!(self.visibility),
            },
        )
        .tile_in::<E>(partitioning)
    }

    /// The bias over `partitioning`'s heads, keys and queries, the keys before the queries: as a
    /// walk owned by the keys holds its scores.
    pub fn keys_first_bias<E: Numeric>(&self, partitioning: &Partitioning) -> Tile {
        Procedural::new::<E, VisibleKeys>(
            comptime!(partitioning.space().subspace(&[G, S, Q])),
            VisibleKeys {
                attended: self.attended,
                seq_q: self.seq_q,
                visibility: comptime!(self.visibility),
            },
        )
        .tile_in::<E>(partitioning)
    }

    /// Steps of `queries_per_step` queries before the first that sees `region`'s first key: what
    /// a walk owned by the keys skips. Under a causal visibility a query sees no key past its own
    /// position, so the queries before `s + queries − attended` see none of the region's; every
    /// query may see a key otherwise.
    pub fn first_query_step(&self, region: &Region, #[comptime] queries_per_step: usize) -> usize {
        let mut first = 0usize;
        if comptime!(self.visibility.is_causal()) {
            let reach = region.origin(S) + self.seq_q as usize;
            if reach > self.attended as usize {
                first = reach - self.attended as usize;
            }
        }
        first / queries_per_step
    }

    /// Steps of `keys_per_step` keys that the first `region_queries` queries of `region` see: the steps from
    /// `region`'s first key on, every `in_turns`-th of them where that many walkers take a walk's
    /// steps in turns. A step the last visible key falls in is taken whole; the bias masks its
    /// tail. A region that starts past every key its queries see, a cube's run of a split walk,
    /// takes none.
    ///
    /// A window bounds the keys a query sees from below too, and the walk still starts at
    /// `region`'s first key: a step wholly before the window weighs nothing in the softmax, which
    /// keeps a masked slice's zero sum.
    pub fn steps(
        &self,
        region: &Region,
        #[comptime] region_queries: usize,
        #[comptime] keys_per_step: usize,
        #[comptime] in_turns: usize,
    ) -> usize {
        let mut seen = self.attended as usize;
        if comptime!(self.visibility.is_causal()) {
            // Query `q` sees key `s` where `s + queries ≤ q + attended`: the region's last query
            // sees below `reach`.
            let reach = region.origin(Q) + region_queries + self.attended as usize;
            if reach > self.seq_q as usize {
                seen = seen.min(reach - self.seq_q as usize);
            } else {
                seen = 0;
            }
        }
        let first = region.origin(S) / keys_per_step;
        let visible = seen.div_ceil(keys_per_step);
        let mut ahead = 0usize;
        if visible > first {
            ahead = visible - first;
        }
        ahead.div_ceil(in_turns)
    }

    /// Whether query `query` sees key `key`.
    pub fn sees(&self, query: u32, key: u32) -> bool {
        let mut seen = key < self.attended;
        if comptime!(self.visibility.is_causal()) {
            seen = seen && key + self.seq_q <= query + self.attended;
        }
        if comptime!(self.visibility.window().is_some()) {
            let window = comptime!(self.visibility.window().unwrap().get() as u32);
            if comptime!(self.visibility.is_causal()) {
                seen = seen && key + self.seq_q + window > query + self.attended;
            } else {
                seen = seen && key + window >= self.attended;
            }
        }
        seen
    }
}

#[cube]
impl<E: Numeric> Recipe<E> for VisibleKeys {
    fn evaluate(&self, coordinates: &RecipeCoords) -> E {
        let seen = self.sees(coordinates.along(Q), coordinates.along(S));
        select(seen, E::from_int(0), E::min_value())
    }
}

impl Reads for VisibleKeysExpand {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        axis == S || (self.visibility.is_causal() && axis == Q)
    }
}
