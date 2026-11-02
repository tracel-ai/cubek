use crate::attention::{AttentionVisibility, B, G, Q, QueryPlacement, S};
use cubecl::tile::procedural::{
    Constant, MemoryValues, Procedural, Reads, Recipe, RecipeCoords, RecipeExpand,
};
use cubecl::tile::space::Coords;
use cubecl::tile::{Axis, Partitioning, Region, Space, Tile};
use cubecl::{self as cubecl, cube, ir::Scope, prelude::*};

/// Which keys a query sees, and the bias its scores are summed with: zero at a key it sees, the
/// minimum value elsewhere. A microkernel that holds a tile of scores sums the bias into it; one
/// that scores keys in its own registers asks [`sees`](VisibleKeys::sees). A query sees the first
/// `attended` keys; under [`Causal`](AttentionVisibility::Causal) none past its own position: query
/// `q` of `queries` sees key `s` where `s + queries ≤ q + queries_end`; under
/// [`Windowed`](AttentionVisibility::Windowed) none `window` or more before it either, where `s +
/// queries + window > q + queries_end`; under [`Recent`](AttentionVisibility::Recent) the last
/// `window` of them, where `s + window ≥ attended`; and, where there is a mask, none whose cell is
/// not zero.
#[derive(CubeType, Clone)]
pub struct VisibleKeys {
    /// Keys attended, a prefix of the buffers': none past it is read.
    pub attended: u32,
    pub seq_q: u32,
    /// The key position just past the last query: the attended keys' end where the queries are
    /// their last positions, `seq_q` where they are the first.
    queries_end: u32,
    /// The mask cut to the owner's sequence and key head: a key whose cell is not zero is hidden.
    mask: ComptimeOption<MemoryValues>,
    #[cube(comptime)]
    pub visibility: AttentionVisibility,
}

#[cube]
impl VisibleKeys {
    /// What the queries of `owner`'s sequence and key head see: its sequence's entry of `len`,
    /// read by stride, so a single length bound at stride zero is every sequence's; `queries` of
    /// them; and the cells of `mask`, over `[B, KV, G, Q, S]`, where there is one.
    pub fn new(
        len: &Tensor<u32>,
        owner: &Region,
        queries: u32,
        mask: &ComptimeOption<Tile>,
        #[comptime] visibility: AttentionVisibility,
    ) -> VisibleKeys {
        let attended = len[owner.origin(B) * len.stride(0)];
        let queries_end = match visibility.queries() {
            Some(QueryPlacement::First) => queries,
            Some(QueryPlacement::Last) | None => attended,
        };

        VisibleKeys {
            attended,
            seq_q: queries,
            queries_end,
            mask: mask.clone().map(|mask| mask.at(owner).values()),
            visibility,
        }
    }

    /// What `queries` see of the first `keys` keys, with no mask: a launch that states the keys
    /// as one count for every sequence, as the backward walks do.
    pub fn of_keys(
        keys: u32,
        queries: u32,
        #[comptime] visibility: AttentionVisibility,
    ) -> VisibleKeys {
        let queries_end = match visibility.queries() {
            Some(QueryPlacement::First) => queries,
            Some(QueryPlacement::Last) | None => keys,
        };

        VisibleKeys {
            attended: keys,
            seq_q: queries,
            queries_end,
            mask: ComptimeOption::new_None(),
            visibility,
        }
    }

    /// The bias the scores are summed with, over `partitioning`'s heads, queries and keys.
    pub fn bias<E: Numeric>(&self, partitioning: &Partitioning) -> Tile {
        self.bias_over::<E>(
            partitioning,
            comptime!(partitioning.space().subspace(&[G, Q, S])),
        )
    }

    /// The bias over `partitioning`'s heads, keys and queries, the keys before the queries: as a
    /// walk owned by the keys holds its scores.
    pub fn keys_first_bias<E: Numeric>(&self, partitioning: &Partitioning) -> Tile {
        self.bias_over::<E>(
            partitioning,
            comptime!(partitioning.space().subspace(&[G, S, Q])),
        )
    }

    /// The bias over `space`, laid out as the scores it is summed into.
    fn bias_over<E: Numeric>(&self, partitioning: &Partitioning, #[comptime] space: Space) -> Tile {
        Procedural::new::<E, VisibleKeys>(
            space,
            VisibleKeys {
                attended: self.attended,
                seq_q: self.seq_q,
                queries_end: self.queries_end,
                mask: self.mask.clone(),
                visibility: comptime!(self.visibility),
            },
        )
        .tile_in::<E>(partitioning)
    }

    /// The biases a walk over `region`'s keys sums its steps' scores with, over `partitioning`'s
    /// heads, queries and keys ([`StepBias`]): none on a step whose every key every query of
    /// `region` sees, the visibility's on any other.
    pub fn step_bias<E: Numeric>(&self, partitioning: &Partitioning, region: &Region) -> StepBias {
        StepBias {
            masked: self.bias::<E>(partitioning),
            unmasked: Procedural::new::<E, Constant<E>>(
                comptime!(partitioning.space().subspace(&[G, Q, S])),
                Constant::<E> {
                    value: E::from_int(0),
                },
            )
            .tile_in::<E>(partitioning),
            seen_by_all: self.seen_by_all(region),
        }
    }

    /// Keys from the first that every query of `region` sees: the attended keys, under a causal
    /// visibility those before the first query's own position too. None under a window or a mask,
    /// which hide keys a count from the first does not say.
    pub fn seen_by_all(&self, region: &Region) -> usize {
        let hides = comptime!(self.visibility.window().is_some() || self.mask.is_some());
        let mut seen = self.attended as usize;
        if comptime!(self.visibility.is_causal()) {
            // Query `q` sees key `s` where `s + queries ≤ q + queries_end`: the region's first
            // query sees below `reach`.
            let reach = region.origin(Q) + 1 + self.queries_end as usize;
            if reach > self.seq_q as usize {
                seen = seen.min(reach - self.seq_q as usize);
            } else {
                seen = 0;
            }
        }

        if comptime!(hides) {
            seen = 0;
        }

        seen
    }

    /// Steps of `queries_per_step` queries before the first that sees `region`'s first key: what
    /// a walk owned by the keys skips. Under a causal visibility a query sees no key past its own
    /// position, so the queries before `s + queries − queries_end` see none of the region's;
    /// every query may see a key otherwise.
    pub fn first_query_step(&self, region: &Region, #[comptime] queries_per_step: usize) -> usize {
        let mut first = 0usize;
        if comptime!(self.visibility.is_causal()) {
            let reach = region.origin(S) + self.seq_q as usize;
            if reach > self.queries_end as usize {
                first = reach - self.queries_end as usize;
            }
        }

        first / queries_per_step
    }

    /// Steps of `keys_per_step` keys that the first `region_queries` queries of `region` see: the
    /// steps from `region`'s first key on, every `in_turns`-th of them where that many walkers take
    /// a walk's steps in turns. A step the last visible key falls in is taken whole; the bias masks
    /// its tail. A region that starts past every key its queries see, a cube's run of a split walk,
    /// takes none.
    ///
    /// A window bounds the keys a query sees from below too, and the walk still starts at
    /// `region`'s first key: a step wholly before the window weighs nothing in the softmax, which
    /// keeps a masked slice's zero sum. A mask is read as the steps are, never to count them.
    pub fn steps(
        &self,
        region: &Region,
        #[comptime] region_queries: usize,
        #[comptime] keys_per_step: usize,
        #[comptime] in_turns: usize,
    ) -> usize {
        let mut seen = self.attended as usize;
        if comptime!(self.visibility.is_causal()) {
            // Query `q` sees key `s` where `s + queries ≤ q + queries_end`: the region's last
            // query sees below `reach`.
            let reach = region.origin(Q) + region_queries + self.queries_end as usize;
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

    /// Whether query `query` of head `head` in its key head's group sees key `key`.
    pub fn sees(&self, head: u32, query: u32, key: u32) -> bool {
        let visibility = comptime!(self.visibility);

        let mut seen = key < self.attended;
        if visibility.is_causal() {
            seen = seen && key + self.seq_q <= query + self.queries_end;
        }

        if visibility.window().is_some() {
            let window = comptime!(visibility.window().unwrap().get() as u32);
            if visibility.is_causal() {
                seen = seen && key + self.seq_q + window > query + self.queries_end;
            } else {
                seen = seen && key + window >= self.attended;
            }
        }

        #[comptime]
        match &self.mask {
            ComptimeOption::None => {}
            ComptimeOption::Some(mask) => {
                let mut cell = Coords::<u32>::new();
                cell.push(head);
                cell.push(query);
                cell.push(key);
                seen = seen && mask.read::<u32>(comptime!(vec![G, Q, S]), &cell) == 0;
            }
        }

        seen
    }
}

#[cube]
impl<E: Numeric> Recipe<E> for VisibleKeys {
    fn evaluate(&self, coordinates: &RecipeCoords) -> E {
        let seen = self.sees(
            coordinates.along(G),
            coordinates.along(Q),
            coordinates.along(S),
        );

        select(seen, E::from_int(0), E::min_value())
    }
}

impl Reads for VisibleKeysExpand {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        let masked = self.mask.is_some();
        axis == S || ((self.visibility.is_causal() || masked) && axis == Q) || (masked && axis == G)
    }
}

/// The biases of a walk's steps ([`VisibleKeys::step_bias`]): a step that lies within the first
/// `seen_by_all` keys, which every query of the walk's region sees, sums its scores with
/// `unmasked`, a constant zero that costs the scores nothing; any other with `masked`, the
/// visibility's, read at each cell. The unmasked steps of a walk in key order come first
/// ([`unmasked_steps`](Self::unmasked_steps)).
#[derive(CubeType, Clone)]
pub struct StepBias {
    pub masked: Tile,
    pub unmasked: Tile,
    pub seen_by_all: usize,
}

#[cube]
impl StepBias {
    /// Both biases windowed to `region`.
    pub fn at(&self, region: &Region) -> StepBias {
        StepBias {
            masked: self.masked.at(region),
            unmasked: self.unmasked.at(region),
            seen_by_all: self.seen_by_all,
        }
    }

    /// Of `steps` steps of `keys` keys from step `first`, those every query sees whole, which
    /// come first: the steps a walk takes with the unmasked bias.
    pub fn unmasked_steps(&self, first: usize, steps: usize, #[comptime] keys: usize) -> usize {
        let seen = self.seen_by_all / keys;
        let mut unmasked = 0usize;
        if seen > first {
            unmasked = steps.min(seen - first);
        }

        unmasked
    }
}
