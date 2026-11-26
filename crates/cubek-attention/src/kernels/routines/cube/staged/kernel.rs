use cubecl::tile::{Partitioning, StageStorage, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, cube_staged_attention_microkernel,
};
use cubek_microkernel::attention::{AttentionVisibility, S, VisibleKeys};

/// Every cube running the cube microkernel on its box with planes that read its stage, laid out
/// as `storage`, over the keys its rows see, and writing each row's logsumexp where the launch binds
/// one.
///
/// - `query`, `key`, `value`, `out`, `logsumexp`, `mask`, `sinks`, `relative`, `len`, `keys`,
///   `scale`: as `unit_attention_kernel`'s.
/// - `query_ends`: the cumulative counts of the sequences' queries, `[b]`, where they lie end to
///   end.
/// - `queries`: the queries a sequence has, or, where they lie end to end, all of them, which caps
///   their ends.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "the query and keys' stage element is defined from them at launch, which the kernel \
              is compiled for"
)]
#[cube(launch_unchecked)]
pub fn cube_staged_attention_kernel<EA: Float, ES: Float, EV: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    logsumexp: ComptimeOption<&mut TileTensor<'static>>,
    mask: ComptimeOption<&TileTensor<'static>>,
    sinks: ComptimeOption<&TileTensor<'static>>,
    relative: ComptimeOption<&TileTensor<'static>>,
    len: &Tensor<u32>,
    query_ends: ComptimeOption<&Tensor<u32>>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] storage: StageStorage,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES, EV)] _elements: [ElemType; 3],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let sinks = sinks.map(|sinks| sinks.tile(&partitioning));
    let relative = relative.map(|relative| relative.tile(&partitioning));

    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);

    let o = out.tile_mut(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile_mut(&partitioning));

    for sequence in &partitioning {
        for cube in sequence {
            let visible =
                VisibleKeys::new(len, &query_ends, &cube, queries, keys, &mask, visibility)
                    .with_terms(&sinks, &relative, &cube);
            let k = key.within(S, 0, visible.attended as usize);
            let v = value.within(S, 0, visible.attended as usize);
            let bias = visible.step_bias::<EA>(&partitioning, &cube);
            let steps = visible.steps(&cube, microkernel.seq_q(), microkernel.seq_kv(), 1usize);

            cube_staged_attention_microkernel::<EA, EV>(
                &cube,
                &o.at(&cube),
                &lse.clone().map(|lse| lse.at(&cube)),
                &q.at(&cube),
                &k.at(&cube),
                &v.at(&cube),
                &bias.at(&cube),
                steps,
                EA::cast_from(scale),
                microkernel,
                storage.clone(),
            );
        }
    }
}
