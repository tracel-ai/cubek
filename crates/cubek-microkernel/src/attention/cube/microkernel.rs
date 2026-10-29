use super::base::CubeAttentionMicrokernel;
use super::plane::CubeAttentionPlane;
use crate::attention::plane::{
    PlaneFragmentAttention, PlaneFragmentAttentionExpand, PlaneShuffleAttention,
    PlaneShuffleAttentionExpand, StepReads, StepReadsExpand,
};
use crate::attention::{B, DV, G, KV, Q, S, StepBias, VisibleKeys};
use crate::split_keys_merge::plane::{
    plane_merge_rows_microkernel, plane_merge_rows_partial_microkernel,
    plane_merge_slot_rows_microkernel,
};
use crate::split_keys_merge::{SlotBox, SlotShape};
use cubecl::tile::layout::PageEntry;
use cubecl::tile::{Region, StageStorage, Stages, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// A cube's planes, fragments or shuffles, as the cube leaves their rows in slots once the walk
/// ends.
#[cube]
trait CubeSlotPlanes<EA: Float, N: Size>: CubeType {
    /// `plane`'s own rows of the region of `walk`'s first step left undivided in slots, the first
    /// `rows` of them: row `r`'s running max, running sum and output at slot `first_slot + r ·
    /// row_stride` of `slot_max`, `slot_sum` and `slot_out`, laid out as
    /// [`SlotShape`](crate::split_keys_merge::SlotShape) states. A plane meets no other here:
    /// planes that split the same rows' keys are merged from their slots by the level above.
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        walk: &Walk,
        plane: &Region,
        slot_out: &mut [Vector<EA, N>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    );
}

#[cube]
impl<EA: Float, ES: Float> CubeSlotPlanes<EA, Const<1>> for PlaneFragmentAttention<EA, ES> {
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        walk: &Walk,
        plane: &Region,
        slot_out: &mut [Vector<EA, Const<1>>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    ) {
        self.store_partial(
            walk, plane, slot_out, slot_max, slot_sum, first_slot, row_stride, rows,
        );
    }
}

#[cube]
impl<EA: Float, N: Size> CubeSlotPlanes<EA, N> for PlaneShuffleAttention<EA, N> {
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        walk: &Walk,
        plane: &Region,
        slot_out: &mut [Vector<EA, N>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    ) {
        self.store_partial(
            walk, plane, slot_out, slot_max, slot_sum, first_slot, row_stride, rows,
        );
    }
}

/// One cube's box: its keys and values walked with planes that read its stage
/// (`staged_walk`), then each plane's rows normalized and drained into the cube's
/// output, which sums the planes that split the same rows' keys, and each row's logsumexp written
/// where the caller asks for it.
///
/// `cube` is the region the cube owns, its walk the stages of keys, of which it takes the first
/// `steps`, each held as `storage`; `out`, `logsumexp`, `query`, `key`, `value` and `bias` are
/// windowed to it.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_microkernel<EA: Float, ES: Float>(
    cube: &Region,
    out: &Tile,
    logsumexp: &ComptimeOption<Tile>,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &StepBias,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] storage: StageStorage,
) {
    let unmasked = bias.unmasked_steps(0, steps, microkernel.seq_kv());
    let mut planes = CubeAttentionMicrokernel::staged_walk::<EA, ES>(
        ComptimeOption::new_Some(cube.walk().range(0, unmasked)),
        cube.walk().range(unmasked, steps - unmasked),
        out,
        query,
        key,
        value,
        bias,
        scale,
        microkernel,
        storage,
    );

    planes.store_normalized(cube, out);
    #[comptime]
    match logsumexp {
        ComptimeOption::Some(logsumexp) => planes.store_logsumexp(cube, logsumexp),
        ComptimeOption::None => {}
    }
}

/// One cube's box, its planes reading their keys and values where they lie
/// (`unstaged_walk`), then each plane's rows normalized and written into the cube's output.
/// Planes that split the same rows' keys leave their rows in slots of shared memory instead
/// (`split_planes_in_slots`), and each merges its share of them into the output
/// (`plane_merge_rows_microkernel`).
///
/// Each row's logsumexp is written where the caller asks for it, by the plane that writes the row.
///
/// `cube` is the region the cube owns, its walk the steps of keys or the pages of them, of which
/// it takes the first `steps`; `out`, `logsumexp` and `query` are windowed to it.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_unstaged_attention_microkernel<
    EA: Float,
    E: Float,
    N: Size,
    K: StepReads<N>,
    L: StepReads<N>,
>(
    cube: &Region,
    out: &Tile,
    logsumexp: &ComptimeOption<Tile>,
    query: &Tile,
    key: &K,
    value: &L,
    visible: &VisibleKeys,
    steps: usize,
    scale: EA,
    #[comptime] value_dim: usize,
    #[comptime] microkernel: CubeAttentionMicrokernel,
) {
    let mut planes = CubeAttentionMicrokernel::unstaged_walk::<EA, E, N, K, L>(
        &cube.walk(),
        steps,
        query,
        key,
        value,
        visible,
        scale,
        value_dim,
        microkernel,
    );

    if microkernel.settings.planes_along_keys.get() == 1 {
        planes.store_normalized::<E>(cube, out);
        #[comptime]
        match logsumexp {
            ComptimeOption::Some(logsumexp) => planes.store_logsumexp(cube, logsumexp),
            ComptimeOption::None => {}
        }
    } else {
        let shape = microkernel.merge_shape(value_dim);
        let (mut slot_out, mut slot_max, mut slot_sum) =
            CubeAttentionMicrokernel::split_planes_in_slots::<EA, N, PlaneShuffleAttention<EA, N>>(
                &mut planes,
                &cube.walk(),
                microkernel,
                shape,
            );
        let queries = microkernel.plane.seq_q();
        let layout = microkernel.layout();
        for plane in cube.planes() {
            let group = plane.coord(G) * layout.planes_along_queries + plane.coord(Q);
            plane_merge_rows_microkernel::<EA, E, N>(
                &mut slot_out,
                &mut slot_max,
                &mut slot_sum,
                &out.at(&plane),
                &logsumexp.clone().map(|logsumexp| logsumexp.at(&plane)),
                group * queries,
                plane.coord(S),
                shape,
                queries,
            );
        }
    }
}

/// One cube's part of its box: its run of the keys walked with planes that read their keys where
/// they lie (`unstaged_walk`), then each plane's partial left in its slot
/// (`store_plane_partials`): what a cube sharing its box with other cubes runs. Returns the box as
/// the slots hold it.
///
/// `run` is the cube level's walk, this cube's run of the steps of keys or the pages of them, of
/// which it takes the first `steps`. `query`, which spans no key, is windowed to the box; `key`
/// and `value` are where each step's lines come from ([`StepReads`]), `value_dim` elements of a
/// value.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_unstaged_attention_partial_microkernel<
    EA: Float,
    E: Float,
    N: Size,
    K: StepReads<N>,
    L: StepReads<N>,
>(
    run: Walk,
    query: &Tile,
    key: &K,
    value: &L,
    visible: &VisibleKeys,
    steps: usize,
    scale: EA,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] splits: usize,
    #[comptime] value_dim: usize,
) -> SlotBox {
    let mut planes = CubeAttentionMicrokernel::unstaged_walk::<EA, E, N, K, L>(
        &run,
        steps,
        query,
        key,
        value,
        visible,
        scale,
        value_dim,
        microkernel,
    );

    // The cube's run starts at its step, or page, of the first `splits`: the cubes take them in
    // turns.
    let cube_slot = run.region(0).origin(S) / microkernel.region_keys();
    CubeAttentionMicrokernel::store_plane_partials::<EA, N, PlaneShuffleAttention<EA, N>>(
        &mut planes,
        &run,
        cube_slot,
        slot_out,
        slot_max,
        slot_sum,
        microkernel,
        splits,
        value_dim,
    )
}

/// One cube's part of its box: its run of the keys walked with planes that read the cube's stage
/// (`staged_walk`), then each plane's partial left in its slot (`store_plane_partials`): what a
/// cube sharing its box with other cubes runs. Returns the box as the slots hold it.
///
/// `staged` and `run` are both the cube level's walk, this cube's run of the steps of keys: the
/// first, of which the ring takes the first `steps`, is spent on the staging, and the second is
/// read for where the slots lie. `out` and `query`, which span no key, are windowed to the box;
/// `key`, `value` and `bias` are read at the walk's regions, each stage of them held as `storage`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_partial_microkernel<EA: Float, ES: Float>(
    staged: Walk,
    run: &Walk,
    out: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &StepBias,
    steps: usize,
    scale: EA,
    slot_out: &mut Tensor<Vector<EA, Const<1>>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] splits: usize,
    #[comptime] storage: StageStorage,
) -> SlotBox {
    // The cubes take the steps in turns: the run's steps every row sees whole are no prefix of
    // it, so every step takes the visibility's bias.
    let mut planes = CubeAttentionMicrokernel::staged_walk::<EA, ES>(
        ComptimeOption::new_None(),
        staged.range(0, steps),
        out,
        query,
        key,
        value,
        bias,
        scale,
        microkernel,
        storage,
    );

    // The cube's run starts at its step of the first `splits`: the cubes take them in turns.
    let cube_slot = run.region(0).origin(S) / microkernel.region_keys();
    CubeAttentionMicrokernel::store_plane_partials::<EA, Const<1>, PlaneFragmentAttention<EA, ES>>(
        &mut planes,
        run,
        cube_slot,
        slot_out,
        slot_max,
        slot_sum,
        microkernel,
        splits,
        comptime!(out.space().extent(DV)),
    )
}

/// One cube's run of a box whose keys several cubes may walk in runs that follow each other: the
/// cube walks its run with planes that read its stage (`staged_walk`), then, where it is the box's
/// only holder, normalizes its planes' rows into the box's output, and each row's logsumexp where
/// the caller asks for it, as [`cube_staged_attention_microkernel()`] does. Otherwise each plane
/// leaves its partial in the box's slot `holder`, the cube's place among the box's `holders`
/// (`store_plane_partials`), and the cube counts itself in at the box's counter of `arrivals`;
/// the last to count merges every holder's slots into the box's output, normalized, with the
/// logsumexp where asked, then leaves the counter at zero.
///
/// The cube takes `steps` steps of `cube`'s walk from step `first`, and `run` is that run, read
/// for where the slots lie. `out`, `logsumexp` and `query`, which span no key, are
/// windowed to `cube`, the box; `key`, `value` and `bias` are read at the walk's regions, each
/// stage of them held as `storage`. A run of no step leaves an empty partial, which merges to
/// nothing. No cube waits on another: the merge does not depend on how the device schedules them.
/// The walk is one, whichever end it takes, so the cube holds its stages once.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_shared_microkernel<EA: Float, ES: Float, E: Float>(
    cube: &Region,
    first: usize,
    steps: usize,
    run: &Walk,
    out: &Tile,
    logsumexp: &ComptimeOption<Tile>,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &StepBias,
    scale: EA,
    slot_out: &mut Tensor<Vector<EA, Const<1>>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    counter: usize,
    holder: usize,
    holders: usize,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] splits: usize,
    #[comptime] storage: StageStorage,
) {
    let value_dim = comptime!(out.space().extent(DV));
    let unmasked = bias.unmasked_steps(first, steps, microkernel.seq_kv());
    let mut planes = CubeAttentionMicrokernel::staged_walk::<EA, ES>(
        ComptimeOption::new_Some(cube.walk().range(first, unmasked)),
        cube.walk().range(first + unmasked, steps - unmasked),
        out,
        query,
        key,
        value,
        bias,
        scale,
        microkernel,
        storage,
    );

    if holders == 1 {
        planes.store_normalized(cube, out);
        #[comptime]
        match logsumexp {
            ComptimeOption::Some(logsumexp) => planes.store_logsumexp(cube, logsumexp),
            ComptimeOption::None => {}
        }
    } else {
        CubeAttentionMicrokernel::store_plane_partials::<
            EA,
            Const<1>,
            PlaneFragmentAttention<EA, ES>,
        >(
            &mut planes,
            run,
            holder,
            slot_out,
            slot_max,
            slot_sum,
            microkernel,
            splits,
            value_dim,
        );

        // The partial lands at device scope before the cube counts itself in.
        sync_storage();

        let mut arrived = Shared::<[u32]>::new_slice(1usize);
        if UNIT_POS == 0 {
            arrived[0] = arrivals[counter].fetch_add(1u32);
        }

        sync_cube();

        // Uniform across the cube: only the box's last cube merges.
        if arrived[0] as usize + 1 == holders {
            // What every other holder stored is acquired before it is read.
            sync_storage();

            let shape = comptime!(SlotShape {
                units: microkernel.units,
                val_dim: value_dim,
                splits,
            });
            let queries = microkernel.plane.seq_q();
            for plane in cube.planes() {
                let plane_box = SlotBox::new::<EA>(
                    slot_max,
                    plane.origin(B),
                    plane.origin(KV),
                    plane.origin(G),
                    plane.origin(Q),
                    queries,
                );

                plane_merge_slot_rows_microkernel::<EA, E, Const<1>>(
                    slot_out,
                    slot_max,
                    slot_sum,
                    &out.at(&plane),
                    &logsumexp.clone().map(|logsumexp| logsumexp.at(&plane)),
                    plane_box.first_row,
                    plane_box.seq_q,
                    holders,
                    shape,
                    queries,
                );
            }

            if UNIT_POS == 0 {
                arrivals[counter].store(0u32);
            }
        }
    }
}

/// The steps a cube microkernel takes on its way: its walks over the keys, and the slots its
/// planes leave their rows in.
#[cube]
impl CubeAttentionMicrokernel {
    /// A cube's planes over the steps of keys it takes, those every row sees whole in `unmasked`,
    /// where the caller has them, and the rest in `walk`, which follows: its query staged once, its
    /// keys and values a stage at a time through the ring for every plane, each plane taking its
    /// part of every stage into its rows of one head with the plane fragment microkernel. Returns
    /// the planes' rows as the walk leaves them, for the level above to end.
    ///
    /// `out` and `query`, which span no key, are windowed to the cube's box; `key`, `value` and
    /// `bias` are read at the walk's regions, each stage of them held as `storage`, the unmasked
    /// steps summed with no bias ([`StepBias`]). The cube meets on the ring's `sync_cube`, its
    /// planes each on their own `sync_plane`.
    #[allow(clippy::too_many_arguments)]
    fn staged_walk<EA: Float, ES: Float>(
        unmasked: ComptimeOption<Walk>,
        walk: Walk,
        out: &Tile,
        query: &Tile,
        key: &Tile,
        value: &Tile,
        bias: &StepBias,
        scale: EA,
        #[comptime] microkernel: Self,
        #[comptime] storage: StageStorage,
    ) -> PlaneFragmentAttention<EA, ES> {
        // The query spans no key: staged once for every plane, before the walk, laid out as the
        // stages of keys and values are.
        let mut staged_query = query.stage_for(
            &walk,
            comptime!(microkernel.query_storage(&query.space())),
            microkernel.plane.decoded_stage(),
        );

        staged_query.copy_from(query);
        let mut planes = match microkernel.plane {
            CubeAttentionPlane::Fragment(plane) => PlaneFragmentAttention::<EA, ES>::open(
                &walk,
                &staged_query,
                out,
                value,
                &bias.masked,
                plane,
                microkernel.planes(),
            ),
            CubeAttentionPlane::Shuffle(_) => {
                unreachable!("a cube whose planes read their keys where they lie stages nothing")
            }
        };

        sync_cube();

        let in_flight = microkernel.ring.stages_in_flight.get();
        let decoded = microkernel.plane.decoded_stage();
        let mut stages = Stages::smem(&walk, key, value, storage, in_flight, decoded);

        // The steps every row sees whole come first and take no bias, the visibility costing their
        // scores nothing; the rest take the visibility's. Two walks of the one ring, so neither
        // step carries the other's bias.
        let prefetch = microkernel.ring.prefetch;
        #[comptime]
        match unmasked {
            ComptimeOption::Some(unmasked) => {
                stages.walk(unmasked, prefetch, |slot, stage| {
                    slot.consume(|key, value| {
                        for plane in stage {
                            planes.step(&plane, key, value, &bias.unmasked, scale);
                        }
                    });
                });
            }
            ComptimeOption::None => {}
        }

        stages.walk(walk, prefetch, |slot, stage| {
            slot.consume(|key, value| {
                for plane in stage {
                    planes.step(&plane, key, value, &bias.masked, scale);
                }
            });
        });

        // Nothing reads a stage once the walk is over: what the cube declares after it takes their
        // bytes, once every unit is past its last read
        // ([`walk_then_shared_memory`](Self::walk_then_shared_memory)).
        stages.free();
        sync_cube();

        planes
    }

    /// A cube's planes over the first `steps` regions of `walk`, its steps or its pages
    /// ([`Self::over_pages`]), each plane reading its part of every step's keys
    /// and values where they lie into its rows of one head with the plane shuffle. Returns the
    /// planes' rows as the walk leaves them, for the level above to end.
    ///
    /// `query`, which spans no key, is windowed to the cube's box; `key` and `value` are where each
    /// step's lines come from ([`StepReads`]), `value_dim` elements of a value, each row seeing the
    /// keys `visible` admits. The planes meet nowhere.
    #[allow(clippy::too_many_arguments)]
    fn unstaged_walk<EA: Float, E: Float, N: Size, K: StepReads<N>, L: StepReads<N>>(
        walk: &Walk,
        steps: usize,
        query: &Tile,
        key: &K,
        value: &L,
        visible: &VisibleKeys,
        scale: EA,
        #[comptime] value_dim: usize,
        #[comptime] microkernel: Self,
    ) -> PlaneShuffleAttention<EA, N> {
        let mut planes = match microkernel.plane {
            CubeAttentionPlane::Shuffle(plane) => {
                PlaneShuffleAttention::<EA, N>::open::<E>(walk, query, value_dim, plane)
            }
            CubeAttentionPlane::Fragment(_) => {
                unreachable!("a cube whose planes read its stage walks it staged")
            }
        };

        if microkernel.page_steps.is_some() {
            // Each page's keys and values cut once, and the steps its rows see cut from them.
            let rows = microkernel.seq_q();
            let keys = microkernel.seq_kv();
            for p in 0..steps.min(walk.total()) {
                let page = walk.region(p);
                let (page_keys, page_values) = (key.at(&page), value.at(&page));
                let in_page = page.walk();
                let seen = visible
                    .steps(&page, rows, keys, 1usize)
                    .min(in_page.total());
                for i in 0..seen {
                    for plane in in_page.region(i) {
                        planes.step::<K, L>(&plane, &page_keys, &page_values, visible, scale);
                    }
                }
            }
        } else {
            Self::walk_reading_ahead::<EA, N, K, L>(
                &mut planes,
                walk,
                steps,
                key,
                value,
                visible,
                scale,
            );
        }

        planes
    }

    /// `steps` of `walk` run by `planes`, each cut with the page table entry read during the step
    /// before it, so the table load overlaps that step's compute. Keys and values read one page
    /// table, so the key's entry serves both.
    fn walk_reading_ahead<EA: Float, N: Size, K: StepReads<N>, L: StepReads<N>>(
        planes: &mut PlaneShuffleAttention<EA, N>,
        walk: &Walk,
        steps: usize,
        key: &K,
        value: &L,
        visible: &VisibleKeys,
        scale: EA,
    ) {
        let count = steps.min(walk.total());
        if count > 0 {
            let last = walk.total() - 1;
            let mut entry = Self::plane_page_entry::<N, K>(walk, key, 0);
            for i in 0..count {
                let next = Self::plane_page_entry::<N, K>(walk, key, (i + 1).min(last));
                let (keys, values) = (key.with_page_entry(&entry), value.with_page_entry(&entry));
                for plane in walk.region(i) {
                    planes.step::<K, L>(&plane, &keys, &values, visible, scale);
                }

                entry = next;
            }
        }
    }

    /// The entry `key` is cut with at step `index` of `walk`, for the plane this worker runs there.
    fn plane_page_entry<N: Size, K: StepReads<N>>(walk: &Walk, key: &K, index: usize) -> PageEntry {
        let mut entry = PageEntry::unlisted();
        for plane in walk.region(index) {
            entry = key.read_page_entry(&plane);
        }

        entry
    }

    /// Every plane of a cube leaving its queries' partial in the cube's slot, `cube_slot` of
    /// `splits`: planes splitting the same queries' keys merge into it
    /// ([`CubeSlotPlanes::store_partial`]). Returns the box as the slots hold it, `run`'s first
    /// step's. The slots lie as [`SlotBox`] states.
    #[allow(clippy::too_many_arguments)]
    fn store_plane_partials<EA: Float, N: Size, P: CubeSlotPlanes<EA, N>>(
        planes: &mut P,
        run: &Walk,
        cube_slot: usize,
        slot_out: &mut Tensor<Vector<EA, N>>,
        slot_max: &mut Tensor<EA>,
        slot_sum: &mut Tensor<EA>,
        #[comptime] cube: Self,
        #[comptime] splits: usize,
        #[comptime] value: usize,
    ) -> SlotBox {
        let first = run.region(0);
        let slot_box = SlotBox::new::<EA>(
            slot_max,
            first.origin(B),
            first.origin(KV),
            first.origin(G),
            first.origin(Q),
            cube.seq_q(),
        );
        let queries = cube.plane.seq_q();
        if cube.settings.planes_along_keys.get() == 1 {
            for plane in run.planes() {
                let plane_box = SlotBox::new::<EA>(
                    slot_max,
                    plane.origin(B),
                    plane.origin(KV),
                    plane.origin(G),
                    plane.origin(Q),
                    queries,
                );

                planes.store_partial(
                    run,
                    &plane,
                    slot_out,
                    slot_max,
                    slot_sum,
                    plane_box.first_row * splits + cube_slot,
                    splits,
                    plane_box.seq_q,
                );
            }
        } else {
            let shape = cube.merge_shape(value);
            let (mut shared_out, mut shared_max, mut shared_sum) =
                Self::split_planes_in_slots::<EA, N, P>(planes, run, cube, shape);
            let layout = cube.layout();
            for plane in run.planes() {
                let plane_box = SlotBox::new::<EA>(
                    slot_max,
                    plane.origin(B),
                    plane.origin(KV),
                    plane.origin(G),
                    plane.origin(Q),
                    queries,
                );
                let group = plane.coord(G) * layout.planes_along_queries + plane.coord(Q);
                plane_merge_rows_partial_microkernel::<EA, N>(
                    &mut shared_out,
                    &mut shared_max,
                    &mut shared_sum,
                    slot_out,
                    slot_max,
                    slot_sum,
                    group * queries,
                    plane.coord(S),
                    plane_box.first_row * splits + cube_slot,
                    splits,
                    plane_box.seq_q,
                    shape,
                    queries,
                );
            }
        }

        slot_box
    }

    /// Every plane's own rows of `walk` left undivided in slots of shared memory, laid out as
    /// `shape` states, its planes laid out as the cube microkernel lays them: the slots of a group
    /// of planes holding the same rows side by side, one for each plane splitting their keys
    /// ([`CubeSlotPlanes::store_partial`]). Returns the slots, once every plane's partial has
    /// landed, for the planes to merge. A meeting of the cube.
    #[allow(
        clippy::type_complexity,
        reason = "the slots' output, maxima and sums, as every reader of slots takes them"
    )]
    fn split_planes_in_slots<EA: Float, N: Size, P: CubeSlotPlanes<EA, N>>(
        planes: &mut P,
        walk: &Walk,
        #[comptime] microkernel: Self,
        #[comptime] shape: SlotShape,
    ) -> (Shared<[Vector<EA, N>]>, Shared<[EA]>, Shared<[EA]>) {
        let queries = microkernel.plane.seq_q();
        let layout = microkernel.layout();
        let slots = layout.planes() * queries;
        let width = N::value();
        let mut slot_out =
            Shared::<[Vector<EA, N>]>::new_slice(comptime!(slots * (shape.val_dim / width)));
        let mut slot_max = Shared::<[EA]>::new_slice(slots);
        let mut slot_sum = Shared::<[EA]>::new_slice(slots);
        for plane in walk.planes() {
            let group = plane.coord(G) * layout.planes_along_queries + plane.coord(Q);
            planes.store_partial(
                walk,
                &plane,
                &mut slot_out,
                &mut slot_max,
                &mut slot_sum,
                group * queries * shape.splits + plane.coord(S),
                shape.splits,
                queries,
            );
        }

        // Every plane's partial lands before any plane merges a row.
        sync_cube();

        (slot_out, slot_max, slot_sum)
    }
}
