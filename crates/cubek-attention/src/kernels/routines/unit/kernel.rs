use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::unit::{UnitAttentionMicrokernel, unit_attention_microkernel};
use cubek_microkernel::attention::{AttentionVisibility, S, VisibleKeys};

/// Every unit of the launch running its rows, and writing each row's logsumexp where the launch
/// binds one.
///
/// - `query`: the query, `[b, kv, g, q, d]`.
/// - `key`, `value`: the keys and the values, read no further than each sequence's attended length.
/// - `out`: the output, `[b, kv, g, q, dv]`.
/// - `logsumexp`: each query's logsumexp, `[b, kv, g, q]`, where the launch binds one.
/// - `mask`: which keys each query does not see, `[b, kv, g, q, s]`, where the launch binds one.
/// - `len`: each sequence's attended length, `[b]`, read by stride.
/// - `queries`, `keys`: queries and keys a sequence; a length past `keys` attends as `keys`.
/// - `scale`: what each dot product is scaled by before the softmax.
#[cube(launch_unchecked)]
pub fn unit_attention_kernel<EA: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    logsumexp: ComptimeOption<&mut TileTensor<'static>>,
    mask: ComptimeOption<&TileTensor<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitAttentionMicrokernel,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA)] _sum: ElemType,
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));

    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);

    let o = out.tile_mut(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile_mut(&partitioning));

    for unit in &partitioning {
        let visible = VisibleKeys::new(
            len,
            &ComptimeOption::new_None(),
            &unit,
            queries,
            keys,
            &mask,
            visibility,
        );
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(
            &unit,
            microkernel.settings.seq_q.get(),
            microkernel.seq_kv(),
            1usize,
        );

        unit_attention_microkernel::<EA>(
            &unit,
            &o.at(&unit),
            &lse.clone().map(|lse| lse.at(&unit)),
            &q.at(&unit),
            &k.at(&unit),
            &v.at(&unit),
            &bias.at(&unit),
            steps,
            EA::cast_from(scale),
            microkernel,
        );
    }
}
