use super::base::CubeAttentionMicrokernel;
use crate::attention::merge::plane::{
    plane_merge_rows_microkernel, plane_merge_rows_partial_microkernel,
};
use crate::attention::{
    AttentionSlotBox, AttentionSlotShape, B, CubePlaneAttention, CubePlaneAttentionExpand,
    CubePlaneMicrokernel, DV, G, GlobalPlaneOperands, KV, Q, S, StagedPlaneOperands, StepLines,
    StepLinesExpand, VisibleKeys,
};
use cubecl::tile::{Region, StageStorage, Stages, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's box: its keys and values walked with planes that read its stage
/// (`staged_walk`), then each plane's rows normalized and drained into the cube's
/// output, which sums the planes that split the same rows' keys.
///
/// `cube` is the region the cube owns, its walk the stages of keys, of which it takes the first
/// `steps`, each held as `storage`; `out`, `query`, `key`, `value` and `bias` are windowed to it.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_microkernel<
    EA: Float,
    E: Float,
    P: CubePlaneAttention<EA, E, Tile, Tile, Tile, Opening = StagedPlaneOperands>,
>(
    cube: &Region,
    out: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
    #[comptime] storage: StageStorage,
) {
    let mut planes = staged_walk::<EA, E, P>(
        cube.walk().range(0, steps),
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
}

/// A cube's planes over `walk`, the steps of keys it takes: its query staged once, its keys and
/// values a stage at a time through the ring for every plane, each plane taking its part of every
/// stage into its rows of one head with its plane attention `P`. Returns the planes' rows as the
/// walk leaves them, for the level above to end.
///
/// `out` and `query`, which span no key, are windowed to the cube's box; `key`, `value` and `bias`
/// are read at the walk's regions, each stage of them held as `storage`. The cube meets on the
/// ring's `sync_cube`, its planes each on their own `sync_plane`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn staged_walk<
    EA: Float,
    E: Float,
    P: CubePlaneAttention<EA, E, Tile, Tile, Tile, Opening = StagedPlaneOperands>,
>(
    walk: Walk,
    out: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    scale: EA,
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
    #[comptime] storage: StageStorage,
) -> P {
    // The query spans no key: staged once for every plane, before the walk.
    let mut staged_query = query.stage_for(
        &walk,
        StageStorage::Strided,
        comptime!(microkernel.plane.decoded_stage()),
    );
    staged_query.copy_from(query);
    let mut planes = P::open(
        &walk,
        &StagedPlaneOperands {
            query: staged_query,
            out: out.clone(),
            value: value.clone(),
            bias: bias.clone(),
        },
        comptime!(microkernel.plane),
        comptime!(microkernel.layout()),
    );
    sync_cube();
    let mut stages = Stages::smem(
        &walk,
        key,
        value,
        storage,
        comptime!(microkernel.ring.stages_in_flight.get()),
        comptime!(microkernel.plane.decoded_stage()),
    );
    stages.walk(walk, comptime!(microkernel.ring.prefetch), |slot, stage| {
        slot.consume(|key, value| {
            for plane in stage {
                planes.step(&plane, key, value, bias, scale);
            }
        });
    });
    planes
}

/// One cube's box, its planes reading their keys and values where they lie
/// (`unstaged_walk`), then each plane's rows normalized and written into the cube's output.
/// Planes that split the same rows' keys leave their rows in slots of shared memory instead
/// (`split_planes_in_slots`), and each merges its share of them into the output
/// ([`plane_merge_rows_microkernel`]).
///
/// `cube` is the region the cube owns, its walk the steps of keys or the pages of them, of which
/// it takes the first `steps`; `out` and `query` are windowed to it.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_unstaged_attention_microkernel<
    EA: Float,
    E: Float,
    N: Size,
    K: StepLines<N>,
    L: StepLines<N>,
    P: CubePlaneAttention<
            EA,
            E,
            K,
            L,
            VisibleKeys,
            Opening = GlobalPlaneOperands,
            Line = Vector<EA, N>,
        >,
>(
    cube: &Region,
    out: &Tile,
    query: &Tile,
    key: &K,
    value: &L,
    visible: &VisibleKeys,
    steps: usize,
    scale: EA,
    #[comptime] value_dim: usize,
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
) {
    let mut planes = unstaged_walk::<EA, E, N, K, L, P>(
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
    if comptime!(microkernel.settings.planes_along_keys.get() == 1) {
        planes.store_normalized(cube, out);
    } else {
        let shape = comptime!(microkernel.merge_shape(value_dim));
        let (mut slot_out, mut slot_max, mut slot_sum) =
            split_planes_in_slots::<EA, E, N, K, L, VisibleKeys, P>(
                &mut planes,
                &cube.walk(),
                microkernel,
                shape,
            );
        let queries = comptime!(microkernel.plane.seq_q());
        let layout = comptime!(microkernel.layout());
        for plane in cube.planes() {
            let group = plane.coord(G) * comptime!(layout.planes_along_queries) + plane.coord(Q);
            plane_merge_rows_microkernel::<EA, E, N>(
                &mut slot_out,
                &mut slot_max,
                &mut slot_sum,
                &out.at(&plane),
                group * queries,
                plane.coord(S),
                shape,
                queries,
            );
        }
    }
}

/// A cube's planes over the first `steps` regions of `walk`, its steps or its pages
/// ([`CubeAttentionMicrokernel::over_pages`]), each plane reading its part of every step's keys
/// and values where they lie into its rows of one head with its plane attention `P`. Returns the
/// planes' rows as the walk leaves them, for the level above to end.
///
/// `query`, which spans no key, is windowed to the cube's box; `key` and `value` are where each
/// step's lines come from ([`StepLines`]), `value_dim` elements of a value, each row seeing the
/// keys `visible` admits. The planes meet nowhere.
#[cube]
#[allow(clippy::too_many_arguments)]
fn unstaged_walk<
    EA: Float,
    E: Float,
    N: Size,
    K: StepLines<N>,
    L: StepLines<N>,
    P: CubePlaneAttention<EA, E, K, L, VisibleKeys, Opening = GlobalPlaneOperands>,
>(
    walk: &Walk,
    steps: usize,
    query: &Tile,
    key: &K,
    value: &L,
    visible: &VisibleKeys,
    scale: EA,
    #[comptime] value_dim: usize,
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
) -> P {
    let mut planes = P::open(
        walk,
        &GlobalPlaneOperands {
            query: query.clone(),
            val_dim: value_dim,
        },
        comptime!(microkernel.plane),
        comptime!(microkernel.layout()),
    );
    match comptime!(microkernel.page_steps.is_some()) {
        // Each page's keys and values cut once, and the steps its rows see cut from them.
        true => {
            let (rows, keys) = comptime!((microkernel.seq_q(), microkernel.seq_kv()));
            for p in 0..steps.min(walk.total()) {
                let page = walk.region(p);
                let (page_keys, page_values) = (key.at(&page), value.at(&page));
                let in_page = page.walk();
                for i in 0..visible
                    .steps(&page, rows, keys, 1usize)
                    .min(in_page.total())
                {
                    for plane in in_page.region(i) {
                        planes.step(&plane, &page_keys, &page_values, visible, scale);
                    }
                }
            }
        }
        false => {
            for i in 0..steps.min(walk.total()) {
                for plane in walk.region(i) {
                    planes.step(&plane, key, value, visible, scale);
                }
            }
        }
    }
    planes
}

/// One cube's part of its box: its run of the keys walked with planes `P` that read their keys
/// where they lie (`unstaged_walk`), then each plane's partial left in its slot
/// (`store_plane_partials`): what a cube sharing its box with other cubes runs. Returns the box as
/// the slots hold it.
///
/// `run` is the cube level's walk, this cube's run of the steps of keys or the pages of them, of
/// which it takes the first `steps`. `query`, which spans no key, is windowed to the box; `key`
/// and `value` are where each step's lines come from ([`StepLines`]), `value_dim` elements of a
/// value.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_unstaged_attention_partial_microkernel<
    EA: Float,
    E: Float,
    N: Size,
    K: StepLines<N>,
    L: StepLines<N>,
    P: CubePlaneAttention<
            EA,
            E,
            K,
            L,
            VisibleKeys,
            Opening = GlobalPlaneOperands,
            Line = Vector<EA, N>,
        >,
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
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
    #[comptime] splits: usize,
    #[comptime] value_dim: usize,
) -> AttentionSlotBox {
    let mut planes = unstaged_walk::<EA, E, N, K, L, P>(
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
    store_plane_partials::<EA, E, N, K, L, VisibleKeys, P>(
        &mut planes,
        &run,
        slot_out,
        slot_max,
        slot_sum,
        microkernel,
        splits,
        value_dim,
    )
}

/// One cube's part of its box: its run of the keys walked with planes `P` that read the cube's
/// stage (`staged_walk`), then each plane's partial left in its slot (`store_plane_partials`):
/// what a cube sharing its box with other cubes runs. Returns the box as the slots hold it.
///
/// `staged` and `run` are both the cube level's walk, this cube's run of the steps of keys: the
/// first, of which the ring takes the first `steps`, is spent on the staging, and the second is
/// read for where the slots lie. `out` and `query`, which span no key, are windowed to the box;
/// `key`, `value` and `bias` are read at the walk's regions, each stage of them held as `storage`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_partial_microkernel<
    EA: Float,
    E: Float,
    N: Size,
    P: CubePlaneAttention<
            EA,
            E,
            Tile,
            Tile,
            Tile,
            Opening = StagedPlaneOperands,
            Line = Vector<EA, N>,
        >,
>(
    staged: Walk,
    run: &Walk,
    out: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    steps: usize,
    scale: EA,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
    #[comptime] splits: usize,
    #[comptime] storage: StageStorage,
) -> AttentionSlotBox {
    let mut planes = staged_walk::<EA, E, P>(
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
    store_plane_partials::<EA, E, N, Tile, Tile, Tile, P>(
        &mut planes,
        run,
        slot_out,
        slot_max,
        slot_sum,
        microkernel,
        splits,
        comptime!(out.space().extent(DV)),
    )
}

/// Every plane of a cube leaving its queries' partial in the cube's slot, at the cube's position
/// along the keys, one of `splits`: planes splitting the same queries' keys merge into it
/// ([`CubePlaneAttention::store_partial`]). Returns the box as the slots hold it, `run`'s first
/// step's. The slots lie as [`AttentionSlotBox`] states.
#[cube]
#[allow(clippy::too_many_arguments)]
fn store_plane_partials<
    EA: Float,
    E: Float,
    N: Size,
    K: CubeType,
    L: CubeType,
    V: CubeType,
    P: CubePlaneAttention<EA, E, K, L, V, Line = Vector<EA, N>>,
>(
    planes: &mut P,
    run: &Walk,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    #[comptime] cube: CubeAttentionMicrokernel<P::Microkernel>,
    #[comptime] splits: usize,
    #[comptime] value: usize,
) -> AttentionSlotBox {
    let first = run.region(0);
    let slot_box = AttentionSlotBox::new::<EA>(
        slot_max,
        first.origin(B),
        first.origin(KV),
        first.origin(G),
        first.origin(Q),
        comptime!(cube.seq_q()),
    );
    // The cube's run starts at its step, or page, of the first `splits`: the cubes take them in
    // turns.
    let cube_slot = first.origin(S) / comptime!(cube.region_keys());
    let queries = comptime!(cube.plane.seq_q());
    if comptime!(cube.settings.planes_along_keys.get() == 1) {
        for plane in run.planes() {
            let plane_box = AttentionSlotBox::new::<EA>(
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
        let shape = comptime!(cube.merge_shape(value));
        let (mut shared_out, mut shared_max, mut shared_sum) =
            split_planes_in_slots::<EA, E, N, K, L, V, P>(planes, run, cube, shape);
        let layout = comptime!(cube.layout());
        for plane in run.planes() {
            let plane_box = AttentionSlotBox::new::<EA>(
                slot_max,
                plane.origin(B),
                plane.origin(KV),
                plane.origin(G),
                plane.origin(Q),
                queries,
            );
            let group = plane.coord(G) * comptime!(layout.planes_along_queries) + plane.coord(Q);
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

/// Every plane's own rows of `walk` left undivided in slots of shared memory, laid
/// out as `shape` states, its planes laid out as the cube microkernel lays them: the slots of a
/// group of planes holding the same rows side by side, one for each plane splitting their keys
/// ([`CubePlaneAttention::store_partial`]). Returns the slots, once every plane's partial has
/// landed, for the planes to merge. A meeting of the cube.
#[cube]
#[allow(
    clippy::type_complexity,
    reason = "the slots' output, maxima and sums, as every reader of slots takes them"
)]
fn split_planes_in_slots<
    EA: Float,
    E: Float,
    N: Size,
    K: CubeType,
    L: CubeType,
    V: CubeType,
    P: CubePlaneAttention<EA, E, K, L, V, Line = Vector<EA, N>>,
>(
    planes: &mut P,
    walk: &Walk,
    #[comptime] microkernel: CubeAttentionMicrokernel<P::Microkernel>,
    #[comptime] shape: AttentionSlotShape,
) -> (Shared<[Vector<EA, N>]>, Shared<[EA]>, Shared<[EA]>) {
    let queries = comptime!(microkernel.plane.seq_q());
    let layout = comptime!(microkernel.layout());
    let slots = comptime!(layout.planes() * queries);
    let width = N::value();
    let mut slot_out =
        Shared::<[Vector<EA, N>]>::new_slice(comptime!(slots * (shape.val_dim / width)));
    let mut slot_max = Shared::<[EA]>::new_slice(slots);
    let mut slot_sum = Shared::<[EA]>::new_slice(slots);
    for plane in walk.planes() {
        let group = plane.coord(G) * comptime!(layout.planes_along_queries) + plane.coord(Q);
        planes.store_partial(
            walk,
            &plane,
            &mut slot_out,
            &mut slot_max,
            &mut slot_sum,
            group * queries * comptime!(shape.splits) + plane.coord(S),
            comptime!(shape.splits),
            queries,
        );
    }
    // Every plane's partial lands before any plane merges a row.
    sync_cube();
    (slot_out, slot_max, slot_sum)
}
