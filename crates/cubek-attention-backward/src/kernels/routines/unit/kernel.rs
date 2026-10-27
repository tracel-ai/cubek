use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::unit::{
    UnitKeyGradientMicrokernel, unit_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::query_gradient::unit::{
    UnitQueryGradientMicrokernel, unit_query_gradient_microkernel,
};

/// Every unit of the launch running its queries: their dot product of the output with its
/// gradient, and their gradient, over the keys they see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn unit_query_gradient_kernel<EA: Float>(
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
    #[comptime] microkernel: UnitQueryGradientMicrokernel,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA)] _sum: ElemType,
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
    for unit in &partitioning {
        let steps = visible.steps(
            &unit,
            comptime!(microkernel.seq_q()),
            comptime!(microkernel.seq_kv()),
            1usize,
        );
        unit_query_gradient_microkernel::<EA>(
            &unit,
            &d_q.at(&unit),
            &delta.at(&unit),
            &q.at(&unit),
            &k.at(&unit),
            &v.at(&unit),
            &o.at(&unit),
            &d_o.at(&unit),
            &lse.at(&unit),
            &bias.at(&unit),
            steps,
            EA::cast_from(scale),
            microkernel,
        );
    }
}

/// Every unit of the launch running its keys: their gradient and their values', over every query
/// of the group's heads that sees them.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn unit_key_gradient_kernel<EA: Float>(
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
    #[comptime] microkernel: UnitKeyGradientMicrokernel,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA)] _sum: ElemType,
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
    for unit in &partitioning {
        let first_step = visible.first_query_step(&unit, comptime!(microkernel.seq_q()));
        unit_key_gradient_microkernel::<EA>(
            &unit,
            &d_k.at(&unit),
            &d_v.at(&unit),
            &q.at(&unit),
            &k.at(&unit),
            &v.at(&unit),
            &d_o.at(&unit),
            &lse.at(&unit),
            &delta.at(&unit),
            &bias.at(&unit),
            first_step,
            EA::cast_from(scale),
            microkernel,
        );
    }
}
