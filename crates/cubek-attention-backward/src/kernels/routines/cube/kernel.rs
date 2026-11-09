use cubecl::ir::ElemType;
use cubecl::tile::launch::Output;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, cube_staged_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::query_gradient::cube::{
    CubeQueryGradientMicrokernel, cube_staged_query_gradient_microkernel,
};
use cubek_std::microkernel::StageRing;

/// Every cube of the launch running its queries through stages it fills for all its planes: their
/// dot product of the output with its gradient, and their gradient, over the keys they see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_query_gradient_kernel<EA: Float, ES: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    logsumexp: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    query_gradient: &TileTensor<'_>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeQueryGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let (q, k, v) = (
        query.tile(&partitioning),
        key.tile(&partitioning),
        value.tile(&partitioning),
    );
    let (o, d_o) = (out.tile(&partitioning), out_gradient.tile(&partitioning));
    let (lse, delta) = (logsumexp.tile(&partitioning), delta.tile(&partitioning));
    let d_q = query_gradient.tile(&partitioning);

    let visible = VisibleKeys::of_keys(keys, queries, visibility);
    let bias = visible.bias::<EA>(&partitioning);

    for cube in &partitioning {
        let steps = visible.steps(&cube, microkernel.seq_q(), microkernel.seq_kv(), 1usize);

        cube_staged_query_gradient_microkernel::<EA, ES>(
            &cube,
            &d_q.at(&cube),
            &delta.at(&cube),
            &q.at(&cube),
            &k.at(&cube),
            &v.at(&cube),
            &o.at(&cube),
            &d_o.at(&cube),
            &lse.at(&cube),
            &bias.at(&cube),
            steps,
            EA::cast_from(scale),
            microkernel,
            ring,
        );
    }
}

/// Every cube of the launch running its keys through stages it fills for all its planes: their
/// gradient and their values', over every query of the group's heads that sees them.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_key_gradient_kernel<EA: Float, ES: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    logsumexp: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    key_gradient: &TileTensor<'_>,
    value_gradient: &TileTensor<'_>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    cube_staged_key_gradient_walks::<EA, ES>(
        &query.tile(&partitioning),
        &key.tile(&partitioning),
        &value.tile(&partitioning),
        &out_gradient.tile(&partitioning),
        &logsumexp.tile(&partitioning),
        &delta.tile(&partitioning),
        &key_gradient.tile(&partitioning),
        &value_gradient.tile(&partitioning),
        queries,
        keys,
        scale,
        &partitioning,
        microkernel,
        ring,
        visibility,
    );
}

/// [`cube_staged_key_gradient_kernel`] with the group's heads split over cubes: every cube adds
/// its heads' part of the keys' and values' gradients into `key_gradient` and `value_gradient`,
/// which every cube of the key head's group adds into.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_key_gradient_sum_kernel<ES: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    logsumexp: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    key_gradient: &Output<'_, f32>,
    value_gradient: &Output<'_, f32>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(ES)] _element: ElemType,
) {
    cube_staged_key_gradient_walks::<f32, ES>(
        &query.tile(&partitioning),
        &key.tile(&partitioning),
        &value.tile(&partitioning),
        &out_gradient.tile(&partitioning),
        &logsumexp.tile(&partitioning),
        &delta.tile(&partitioning),
        &key_gradient.tile(&partitioning),
        &value_gradient.tile(&partitioning),
        queries,
        keys,
        scale,
        &partitioning,
        microkernel,
        ring,
        visibility,
    );
}

/// Every cube's walk of its keys, the gradients written into `key_gradient` and
/// `value_gradient`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn cube_staged_key_gradient_walks<EA: Float, ES: Float>(
    q: &Tile,
    k: &Tile,
    v: &Tile,
    d_o: &Tile,
    lse: &Tile,
    delta: &Tile,
    key_gradient: &Tile,
    value_gradient: &Tile,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: &Partitioning,
    #[comptime] microkernel: CubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
) {
    let visible = VisibleKeys::of_keys(keys, queries, visibility);
    let bias = visible.keys_first_bias::<EA>(partitioning);
    let no_share = ComptimeOption::new_None();

    for cube in partitioning {
        let first_step = visible.first_query_step(&cube, microkernel.seq_q());
        cube_staged_key_gradient_microkernel::<EA, ES>(
            &cube,
            &key_gradient.at(&cube),
            &value_gradient.at(&cube),
            &no_share,
            &q.at(&cube),
            &k.at(&cube),
            &v.at(&cube),
            &d_o.at(&cube),
            &lse.at(&cube),
            &delta.at(&cube),
            &bias.at(&cube),
            first_step,
            EA::cast_from(scale),
            microkernel,
            ring,
        );
    }
}
