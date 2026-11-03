use cubecl::ir::ElemType;
use cubecl::tile::launch::Output;
use cubecl::tile::{Partitioning, Tile, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};
use cubek_microkernel::attention_backward::delta::plane::plane_delta_microkernel;
use cubek_microkernel::attention_backward::key_gradient::cube::{
    QueryGradientShare, cube_staged_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::key_gradient::hypercube::HypercubeKeyGradientMicrokernel;
use cubek_std::microkernel::StageRing;

/// Every plane of the launch writing its queries' dot product of the output with its gradient.
#[cube(launch_unchecked)]
pub fn plane_delta_kernel<EA: Float>(
    out: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    delta: &TileArg<'_>,
    partitioning: Partitioning,
    #[define(EA)] _sum: ElemType,
) {
    let (o, d_o) = (out.tile(&partitioning), out_gradient.tile(&partitioning));
    let delta = delta.tile(&partitioning);

    for plane in &partitioning {
        plane_delta_microkernel::<EA>(&plane, &delta.at(&plane), &d_o.at(&plane), &o.at(&plane));
    }
}

/// Every cube of the launch running its keys through stages of its own: their gradient and their
/// values', and its share of the query's gradient added into `query_gradient`, which every cube
/// that sees those queries adds into.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_key_gradient_kernel<EA: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    delta: &TileArg<'_>,
    key_gradient: &TileArg<'_>,
    value_gradient: &TileArg<'_>,
    query_gradient: &Output<'_, f32>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    hypercube_key_gradient_walks::<EA, ES>(
        &query.tile(&partitioning),
        &key.tile(&partitioning),
        &value.tile(&partitioning),
        &out_gradient.tile(&partitioning),
        &logsumexp.tile(&partitioning),
        &delta.tile(&partitioning),
        &key_gradient.tile(&partitioning),
        &value_gradient.tile(&partitioning),
        &query_gradient.tile(&partitioning),
        queries,
        keys,
        scale,
        &partitioning,
        microkernel,
        ring,
        visibility,
    );
}

/// [`hypercube_key_gradient_kernel`] with the group's heads split over cubes: every cube also adds
/// its heads' part of the keys' and values' gradients into `key_gradient` and `value_gradient`,
/// which every cube of the key head's group adds into.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_key_gradient_sum_kernel<ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    delta: &TileArg<'_>,
    key_gradient: &Output<'_, f32>,
    value_gradient: &Output<'_, f32>,
    query_gradient: &Output<'_, f32>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(ES)] _element: ElemType,
) {
    hypercube_key_gradient_walks::<f32, ES>(
        &query.tile(&partitioning),
        &key.tile(&partitioning),
        &value.tile(&partitioning),
        &out_gradient.tile(&partitioning),
        &logsumexp.tile(&partitioning),
        &delta.tile(&partitioning),
        &key_gradient.tile(&partitioning),
        &value_gradient.tile(&partitioning),
        &query_gradient.tile(&partitioning),
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
/// `value_gradient`, and its share of the query's added into `query_gradient`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn hypercube_key_gradient_walks<EA: Float, ES: Float>(
    q: &Tile,
    k: &Tile,
    v: &Tile,
    d_o: &Tile,
    lse: &Tile,
    delta: &Tile,
    key_gradient: &Tile,
    value_gradient: &Tile,
    query_gradient: &Tile,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: &Partitioning,
    #[comptime] microkernel: HypercubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
) {
    let visible = VisibleKeys::of_keys(keys, queries, visibility);
    let bias = visible.keys_first_bias::<EA>(partitioning);

    for cube in partitioning {
        let first_step = visible.first_query_step(&cube, microkernel.seq_q());
        let share = ComptimeOption::new_Some(QueryGradientShare {
            sum: query_gradient.at(&cube),
            product: microkernel.query_gradient,
        });

        cube_staged_key_gradient_microkernel::<EA, ES>(
            &cube,
            &key_gradient.at(&cube),
            &value_gradient.at(&cube),
            &share,
            &q.at(&cube),
            &k.at(&cube),
            &v.at(&cube),
            &d_o.at(&cube),
            &lse.at(&cube),
            &delta.at(&cube),
            &bias.at(&cube),
            first_step,
            EA::cast_from(scale),
            microkernel.cube,
            ring,
        );
    }
}

/// Every plane of the launch casting its queries' gradient out of the sum every cube added into.
#[cube(launch_unchecked)]
pub fn cast_query_gradient_kernel(
    sum: &TileArg<'_>,
    query_gradient: &TileArg<'_>,
    partitioning: Partitioning,
) {
    let (sum, d_q) = (sum.tile(&partitioning), query_gradient.tile(&partitioning));

    for plane in &partitioning {
        let mut d_q = d_q.at(&plane);
        d_q.copy_cast_from(&sum.at(&plane));
    }
}
