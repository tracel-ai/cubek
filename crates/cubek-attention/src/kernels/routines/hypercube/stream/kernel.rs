use cubecl::tile::{Partitioning, StageStorage, Tile, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::cube_staged_attention_shared_microkernel;
use cubek_microkernel::attention::hypercube::HypercubeAttentionMicrokernel;
use cubek_microkernel::attention::{AttentionVisibility, S, VisibleKeys};

/// Every cube walking its run of the line every box's steps lie on, end to end: for each box the
/// run touches, its part of the box's keys, with plane fragments that read its stage. A box the
/// run holds whole is written into the output as the cube routine writes one; a part of a box
/// several runs share is left in the slot of the run's place among them, and the box's last run
/// to count itself in merges them into the output
/// ([`cube_staged_attention_shared_microkernel()`]). Each row's logsumexp is written where the
/// launch binds one.
///
/// The line holds every box's attended steps, as many as the longest sequence's: the steps a box's
/// rows do not see, past a shorter sequence's keys or under a causal mask, are walked by no one,
/// so every run but the last holds the same steps whatever the keys. A cube whose run is empty
/// touches no box and leaves at once.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_stream_fragment_attention_kernel<EA: Float, E: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    logsumexp: ComptimeOption<&TileArg<'static>>,
    slot_out: &mut Tensor<Vector<EA, Const<1>>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    mask: ComptimeOption<&TileArg<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeAttentionMicrokernel,
    #[comptime] splits: usize,
    #[comptime] visibility: AttentionVisibility,
    #[comptime] storage: StageStorage,
    #[define(EA, E, ES)] _elements: [ElemType; 3],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));
    let keys_per_step = microkernel.cube.seq_kv();

    // The longest sequence's attended steps, a box's share of the line; one at least, so a box of
    // no key is still some run's, which leaves it empty.
    let mut longest = 1usize;

    for b in 0..len.shape(0) {
        longest = longest.max((len[b * len.stride(0)] as usize).div_ceil(keys_per_step));
    }

    let portion = partitioning.walk().portion_of(longest);

    for i in 0..portion.touched() {
        let cube = portion.region(i);
        let (from, steps) = portion.steps(i);
        let (first, holders) = portion.holders(i);
        let visible = VisibleKeys::new(len, &cube, queries, &mask, visibility);
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.step_bias::<EA>(&partitioning, &cube);

        // The run's steps of the box its rows see: none past them is walked.
        let seen = visible.steps(&cube, microkernel.cube.seq_q(), keys_per_step, 1usize);

        let mut taken = 0usize;
        if seen > from {
            taken = steps.min(seen - from);
        }

        let run = cube.walk().range(from, taken);
        cube_staged_attention_shared_microkernel::<EA, ES, E>(
            &cube,
            from,
            taken,
            &run,
            &o.at(&cube),
            &lse.clone().map(|lse| lse.at(&cube)),
            &q.at(&cube),
            &k.at(&cube),
            &v.at(&cube),
            &bias.at(&cube),
            EA::cast_from(scale),
            slot_out,
            slot_max,
            slot_sum,
            arrivals,
            portion.index(i),
            portion.run() - first,
            holders,
            microkernel.cube,
            splits,
            storage.clone(),
        );
    }
}
