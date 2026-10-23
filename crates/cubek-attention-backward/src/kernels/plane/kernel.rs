use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::plane::{
    PlaneKeyGradientMicrokernel, plane_fragment_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, plane_fragment_query_gradient_microkernel,
};
use cubek_microkernel::staging::StageRing;

/// Every plane of the launch running its queries through stages of its own: their dot product of
/// the output with its gradient, and their gradient, over the keys they see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_query_gradient_kernel<EA: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    delta: &TileArg<'_>,
    query_gradient: &TileArg<'_>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
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
    let visible = VisibleKeys {
        attended: keys,
        seq_q: queries,
        visibility,
    };
    let bias = visible.bias::<EA>(&partitioning);
    for plane in &partitioning {
        let steps = visible.steps(
            &plane,
            comptime!(microkernel.seq_q()),
            comptime!(microkernel.seq_kv()),
            1usize,
        );
        plane_fragment_query_gradient_microkernel::<EA, ES>(
            &plane,
            &d_q.at(&plane),
            &delta.at(&plane),
            &q.at(&plane),
            &k.at(&plane),
            &v.at(&plane),
            &o.at(&plane),
            &d_o.at(&plane),
            &lse.at(&plane),
            &bias.at(&plane),
            steps,
            EA::cast_from(scale),
            microkernel,
            ring,
            planes,
        );
    }
}

/// Every plane of the launch running its keys through stages of its own: their gradient and
/// their values', over every query of the group's heads that sees them.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_key_gradient_kernel<EA: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    delta: &TileArg<'_>,
    key_gradient: &TileArg<'_>,
    value_gradient: &TileArg<'_>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let (q, k, v) = (
        query.tile(&partitioning),
        key.tile(&partitioning),
        value.tile(&partitioning),
    );
    let d_o = out_gradient.tile(&partitioning);
    let (lse, delta) = (logsumexp.tile(&partitioning), delta.tile(&partitioning));
    let (d_k, d_v) = (
        key_gradient.tile(&partitioning),
        value_gradient.tile(&partitioning),
    );
    let visible = VisibleKeys {
        attended: keys,
        seq_q: queries,
        visibility,
    };
    let bias = visible.keys_first_bias::<EA>(&partitioning);
    for plane in &partitioning {
        let first_step = visible.first_query_step(&plane, comptime!(microkernel.seq_q()));
        plane_fragment_key_gradient_microkernel::<EA, ES>(
            &plane,
            &d_k.at(&plane),
            &d_v.at(&plane),
            &q.at(&plane),
            &k.at(&plane),
            &v.at(&plane),
            &d_o.at(&plane),
            &lse.at(&plane),
            &delta.at(&plane),
            &bias.at(&plane),
            first_step,
            EA::cast_from(scale),
            microkernel,
            ring,
            planes,
        );
    }
}
