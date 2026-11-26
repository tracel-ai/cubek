use crate::attention::{AttentionVisibility, B, G, Q, QueryPlacement, S};
use cubecl::tile::procedural::{
    MemoryValues, Procedural, Reads, Recipe, RecipeCoords, RecipeExpand,
};
use cubecl::tile::{Dimension, Partitioning, Region, Space, Tile};
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
    pub queries_end: u32,
    /// The mask cut to the owner's sequence and key head: a key whose cell is not zero is hidden.
    pub mask: ComptimeOption<MemoryValues>,
    #[cube(comptime)]
    pub visibility: AttentionVisibility,
}

#[cube]
impl VisibleKeys {
    /// What the queries of `owner`'s sequence and key head see: its sequence's attended length
    /// ([`attended_length`](Self::attended_length)); `queries` of them, or, where every sequence's
    /// queries lie end to end, the count of its run of `query_ends`, their cumulative counts, over
    /// `queries` in all; and the cells of `mask`, over `[B, KV, G, Q, S]`, where there is one.
    ///
    /// # Guarantees
    ///
    /// Where the ends never decrease and the last is `queries`, a run holds the queries its ends
    /// count. Whatever `len` and `query_ends` hold, no key at or past `keys` is seen, and a run
    /// holds no query past the `queries` there are: an end is read as at most `queries`, and a
    /// start as at most its end. Where `query_ends` holds an entry, an owner past its entries
    /// reads the last; every launch of this crate has one, a launch of no sequence running no cube.
    #[allow(
        clippy::clone_on_copy,
        reason = "the cube macro expands no dereference of a comptime option, only its clone"
    )]
    pub fn new(
        len: &Tensor<u32>,
        query_ends: &ComptimeOption<&Tensor<u32>>,
        owner: &Region,
        queries: u32,
        keys: u32,
        mask: &ComptimeOption<Tile>,
        #[comptime] visibility: AttentionVisibility,
    ) -> VisibleKeys {
        let sequence = owner.origin(B);
        let attended = VisibleKeys::attended_length(len, sequence, keys);
        let queries = query_ends
            .clone()
            .map::<u32, _>(|ends| {
                let run = VisibleKeys::capped_entry(sequence, ends.len());
                let end = ends[run].min(queries);
                let mut start = 0u32;
                if run > 0 {
                    start = ends[run - 1].min(end);
                }
                end - start
            })
            .unwrap_or(queries);
        // A box that starts past its sequence's run holds no query, so it attends no key.
        let attended = query_ends
            .clone()
            .map::<u32, _>(|_| select(u32::cast_from(owner.origin(Q)) < queries, attended, 0u32))
            .unwrap_or(attended);
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

    /// The keys sequence `sequence` attends: its entry of `len`, read by stride, so a single
    /// length bound at stride zero is every sequence's.
    ///
    /// # Guarantees
    ///
    /// At most `keys`: a length past them attends as a length of the keys there are. Where `len`
    /// holds an entry, a sequence past its entries reads the last; every launch of this crate has
    /// one, a launch of no sequence running no cube.
    pub fn attended_length(len: &Tensor<u32>, sequence: usize, keys: u32) -> u32 {
        let entry = VisibleKeys::capped_entry(sequence, len.shape(0));
        len[entry * len.stride(0)].min(keys)
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

    /// The biases a walk over `region`'s keys sums its steps' scores with, over `partitioning`'s
    /// heads, queries and keys ([`StepBias`]): none on a step whose every key every query of
    /// `region` sees, the visibility's on any other.
    pub fn step_bias<E: Numeric>(&self, partitioning: &Partitioning, region: &Region) -> StepBias {
        StepBias {
            masked: self.bias::<E>(partitioning),
            unmasked: Procedural::zero::<E>(comptime!(partitioning.space().subspace(&[G, Q, S])))
                .tile_in::<E>(partitioning),
            seen_by_all: self.seen_by_all(region),
        }
    }

    /// Keys from the first that every query of `region` sees: the attended keys, under a causal
    /// visibility those before the first query's own position too. None under a window or a mask,
    /// which hide keys a count from the first does not say.
    pub fn seen_by_all(&self, region: &Region) -> usize {
        // Query `q` sees key `s` where `s + queries ≤ q + queries_end`: the region's first query
        // sees below its reach.
        let reach = region.origin(Q) + 1 + self.queries_end as usize;
        let mut seen = self.seen_below(reach);
        if comptime!(self.visibility.window().is_some() || self.mask.is_some()) {
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
            first = VisibleKeys::past(
                region.origin(S) + self.seq_q as usize,
                self.queries_end as usize,
            );
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
        // The region's last query sees below its reach.
        let reach = region.origin(Q) + region_queries + self.queries_end as usize;
        let visible = self.seen_below(reach).div_ceil(keys_per_step);
        VisibleKeys::past(visible, region.origin(S) / keys_per_step).div_ceil(in_turns)
    }

    /// Whether query `query` of head `head` in its key head's group sees key `key`.
    pub fn sees(&self, head: u32, query: u32, key: u32) -> bool {
        key < self.attended && self.placed_sees(query, key) && self.mask_shows(head, query, key)
    }
}

#[cube]
impl VisibleKeys {
    /// `sequence`, a sequence's entry of a tensor holding `entries`, read as the last past them.
    fn capped_entry(sequence: usize, entries: usize) -> usize {
        sequence.min(entries - 1)
    }

    /// How far `value` lies past `start`: none where it does not.
    fn past(value: usize, start: usize) -> usize {
        value.max(start) - start
    }

    /// Keys from the first that a query reaching `reach` sees, of the attended ones: under a
    /// causal visibility, those below its reach past the queries.
    fn seen_below(&self, reach: usize) -> usize {
        let mut seen = self.attended as usize;
        if comptime!(self.visibility.is_causal()) {
            seen = seen.min(VisibleKeys::past(reach, self.seq_q as usize));
        }
        seen
    }

    /// Whether query `query` sees key `key` by their positions: under a causal visibility none
    /// past its own, and under a window none outside it.
    fn placed_sees(&self, query: u32, key: u32) -> bool {
        let mut seen = self.within_window(query, key);
        if comptime!(self.visibility.is_causal()) {
            seen = seen && key + self.seq_q <= query + self.queries_end;
        }
        seen
    }

    /// Whether key `key` lies in query `query`'s window, where the visibility has one: under a
    /// causal visibility fewer than `window` before the query, otherwise among the last `window`
    /// attended.
    fn within_window(&self, query: u32, key: u32) -> bool {
        let window = comptime!(self.visibility.window().map_or(0, |window| window.get()) as u32);
        let mut within = true;
        if comptime!(self.visibility.window().is_some()) {
            within = self.window_holds(query, key, window);
        }
        within
    }

    /// Whether key `key` lies in query `query`'s window of `window` keys.
    fn window_holds(&self, query: u32, key: u32, window: u32) -> bool {
        let mut holds = key + window >= self.attended;
        if comptime!(self.visibility.is_causal()) {
            holds = key + self.seq_q + window > query + self.queries_end;
        }
        holds
    }

    /// Whether the mask, where there is one, leaves key `key` seen by query `query` of head `head`.
    #[allow(
        clippy::clone_on_copy,
        reason = "the cube macro expands no dereference of a comptime option, only its clone"
    )]
    fn mask_shows(&self, head: u32, query: u32, key: u32) -> bool {
        self.mask
            .clone()
            .map::<bool, _>(|mask| VisibleKeys::cell_is_zero(&mask, head, query, key))
            .unwrap_or(true)
    }

    /// Whether `mask`'s cell of head `head`, query `query` and key `key` is zero.
    fn cell_is_zero(mask: &MemoryValues, head: u32, query: u32, key: u32) -> bool {
        let mut cell = Sequence::<u32>::new();
        cell.push(head);
        cell.push(query);
        cell.push(key);
        mask.read::<u32>(comptime!(vec![G, Q, S]), &cell) == 0
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
    fn reads(&self, _scope: &Scope, dimension: Dimension) -> bool {
        let masked = self.mask.is_some();
        dimension == S
            || ((self.visibility.is_causal() || masked) && dimension == Q)
            || (masked && dimension == G)
    }
}

/// The biases of a walk's steps ([`VisibleKeys::step_bias`]): a step that lies within the first
/// `seen_by_all` keys, which every query of the walk's region sees, sums its scores with
/// `unmasked`, zero everywhere, which the softmax takes as no bias ([`Procedural::zero`]): the max
/// over the raw scores, scaled once a row, and each cell's exponential one multiply-add. Any other
/// with `masked`, the visibility's, read at each cell. The unmasked steps of a walk in key order
/// come first ([`unmasked_steps`](Self::unmasked_steps)).
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

    /// Of `steps` steps of `keys` keys taken one every `in_turns` from step `first`, as a cube
    /// sharing its box with `in_turns - 1` others takes them, those every query sees whole: the
    /// steps a walk takes with the unmasked bias. A run's steps rise and only the last of a box
    /// are masked, so they come first.
    pub fn unmasked_steps(
        &self,
        first: usize,
        steps: usize,
        #[comptime] keys: usize,
        #[comptime] in_turns: usize,
    ) -> usize {
        let seen = self.seen_by_all / keys;
        let mut unmasked = 0usize;
        if seen > first {
            unmasked = steps.min((seen - first).div_ceil(in_turns));
        }

        unmasked
    }
}
