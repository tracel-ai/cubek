use cubecl::prelude::*;
use cubecl::tile::{Partitioning, StageStorage, Tile, TileArg};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, cube_staged_attention_microkernel,
};
use cubek_microkernel::attention::{AttentionVisibility, S, VisibleKeys};

/// Every cube running the cube microkernel on its box with planes that read its stage, held as
/// `storage`, over the keys its rows see, and writing each row's logsumexp where the launch binds
/// one.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_kernel<EA: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    logsumexp: ComptimeOption<&TileArg<'static>>,
    mask: ComptimeOption<&TileArg<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] storage: StageStorage,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));
    for cube in &partitioning {
        let visible = VisibleKeys::new(len, &cube, queries, &mask, visibility);
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(&cube, microkernel.seq_q(), microkernel.seq_kv(), 1usize);

        cube_staged_attention_microkernel::<EA, ES>(
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
