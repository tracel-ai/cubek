use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, cube_staged_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::query_gradient::cube::{
    CubeQueryGradientMicrokernel, cube_staged_query_gradient_microkernel,
};
use cubek_microkernel::staging::StageRing;

/// Every cube of the launch running its queries through stages it fills for all its planes: their
/// dot product of the output with its gradient, and their gradient, over the keys they see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_query_gradient_kernel<EA: Float, ES: Float>(
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
    let visible = VisibleKeys {
        attended: keys,
        seq_q: queries,
        visibility,
    };
    let bias = visible.bias::<EA>(&partitioning);
    for cube in &partitioning {
        let steps = visible.steps(
            &cube,
            comptime!(microkernel.seq_q()),
            comptime!(microkernel.seq_kv()),
            1usize,
        );
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
    #[comptime] microkernel: CubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
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
    for cube in &partitioning {
        let first_step = visible.first_query_step(&cube, comptime!(microkernel.seq_q()));
        cube_staged_key_gradient_microkernel::<EA, ES>(
            &cube,
            &d_k.at(&cube),
            &d_v.at(&cube),
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
