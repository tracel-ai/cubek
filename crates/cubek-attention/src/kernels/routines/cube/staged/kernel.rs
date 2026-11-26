use cubecl::tile::{Partitioning, StageStorage, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, cube_staged_attention_microkernel,
};
use cubek_microkernel::attention::{AttentionVisibility, S, VisibleKeys};

/// Every cube running the cube microkernel on its box with planes that read its stage, held as
/// `storage`, over the keys its rows see, and writing each row's logsumexp where the launch binds
/// one.
///
/// - `query`: the query, `[b, kv, g, q, d]`.
/// - `key`, `value`: the keys and the values, read no further than each sequence's attended length.
/// - `out`: the output, `[b, kv, g, q, dv]`.
/// - `logsumexp`: each query's logsumexp, `[b, kv, g, q]`, where the launch binds one.
/// - `mask`: which keys each query does not see, `[b, kv, g, q, s]`, where the launch binds one.
/// - `len`: each sequence's attended length, `[b]`, read by stride.
/// - `query_ends`: the cumulative counts of the sequences' queries, `[b]`, where they lie end to
///   end.
/// - `queries`, `keys`: queries and keys a sequence; a length past `keys` attends as `keys`.
/// - `scale`: what each dot product is scaled by before the softmax.
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

    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);

    let o = out.tile_mut(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile_mut(&partitioning));

    for sequence in &partitioning {
        for cube in sequence {
            let visible =
                VisibleKeys::new(len, &query_ends, &cube, queries, keys, &mask, visibility);
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
