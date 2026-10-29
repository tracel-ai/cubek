use core::marker::PhantomData;

use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::cube_unstaged_attention_partial_microkernel;
use cubek_microkernel::attention::hypercube::HypercubeAttentionMicrokernel;
use cubek_microkernel::attention::plane::{PackedStepReads, TileStepReads};
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};
use cubek_microkernel::split_keys_merge::hypercube::{
    HypercubeMergeShape, hypercube_split_keys_merge_microkernel,
};

/// Every cube walking its run of its box over the keys its queries see, its planes reading the
/// keys and values from tiles, leaving its partial in its slot, then counting itself in at the
/// box, the last merging it ([`hypercube_split_keys_merge_microkernel()`]), with each query's
/// logsumexp where the launch binds one.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_attention_arrival_kernel<EA: Float, E: Float, N: Size>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &mut Tensor<E>,
    logsumexp: ComptimeOption<&mut Tensor<EA>>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    mask: ComptimeOption<&TileArg<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeAttentionMicrokernel,
    #[comptime] merge: HypercubeMergeShape,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let key = TileStepReads::<E> {
        tile: key.tile(&partitioning),
        element: PhantomData,
    };
    let value = TileStepReads::<E> {
        tile: value.tile(&partitioning),
        element: PhantomData,
    };

    // The cube level's walk: this cube's run of the steps, or pages, of keys.
    let run = partitioning.walk();
    let first_step = run.region(0);
    let q = query.tile(&partitioning);
    let visible = VisibleKeys::new(len, &first_step, queries, &mask, visibility);
    let steps = visible.steps(
        &first_step,
        microkernel.cube.seq_q(),
        microkernel.cube.region_keys(),
        merge.cubes,
    );
    let slot_box = cube_unstaged_attention_partial_microkernel::<
        EA,
        E,
        N,
        TileStepReads<E>,
        TileStepReads<E>,
    >(
        run,
        // The query spans no key: its box is the first step's.
        &q.at(&first_step),
        &key,
        &value,
        &visible,
        steps,
        EA::cast_from(scale),
        slot_out,
        slot_max,
        slot_sum,
        microkernel.cube,
        merge.slots.splits,
        merge.slots.val_dim,
    );

    hypercube_split_keys_merge_microkernel::<EA, E, N>(
        &slot_box, slot_out, slot_max, slot_sum, out, logsumexp, arrivals, merge,
    );
}

/// [`hypercube_attention_arrival_kernel`] over a packed cache, its planes decoding the keys and
/// values as they read them ([`PackedStepReads`]).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_attention_arrival_packed_kernel<EA: Float, E: Float, N: Size>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    key_fixup: ComptimeOption<&TileArg<'static>>,
    out: &mut Tensor<E>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    mask: ComptimeOption<&TileArg<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeAttentionMicrokernel,
    #[comptime] merge: HypercubeMergeShape,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let key = PackedStepReads {
        tile: key.tile(&partitioning),
        fixup: key_fixup.map(|fixup| fixup.tile(&partitioning)),
    };
    let value = PackedStepReads {
        tile: value.tile(&partitioning),
        fixup: ComptimeOption::new_None(),
    };

    // The cube level's walk: this cube's run of the steps, or pages, of keys.
    let run = partitioning.walk();
    let first_step = run.region(0);
    let q = query.tile(&partitioning);
    let visible = VisibleKeys::new(len, &first_step, queries, &mask, visibility);
    let steps = visible.steps(
        &first_step,
        microkernel.cube.seq_q(),
        microkernel.cube.region_keys(),
        merge.cubes,
    );
    let slot_box =
        cube_unstaged_attention_partial_microkernel::<EA, E, N, PackedStepReads, PackedStepReads>(
            run,
            // The query spans no key: its box is the first step's.
            &q.at(&first_step),
            &key,
            &value,
            &visible,
            steps,
            EA::cast_from(scale),
            slot_out,
            slot_max,
            slot_sum,
            microkernel.cube,
            merge.slots.splits,
            merge.slots.val_dim,
        );
    // A packed cache is read for inference: no backward pass reads its logsumexp.
    hypercube_split_keys_merge_microkernel::<EA, E, N>(
        &slot_box,
        slot_out,
        slot_max,
        slot_sum,
        out,
        ComptimeOption::new_None(),
        arrivals,
        merge,
    );
}
