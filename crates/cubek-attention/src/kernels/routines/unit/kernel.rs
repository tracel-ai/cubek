use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::unit::{UnitAttentionMicrokernel, unit_attention_microkernel};
use cubek_microkernel::attention::{AttentionVisibility, B, S, VisibleKeys};

/// Every unit of the launch running its rows.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn unit_attention_kernel<EA: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitAttentionMicrokernel,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA)] _sum: ElemType,
) {
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    for unit in &partitioning {
        let visible = VisibleKeys::new(len, unit.origin(B), queries, visibility);
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(
            &unit,
            comptime!(microkernel.settings.seq_q.get()),
            comptime!(microkernel.seq_kv()),
            1usize,
        );
        unit_attention_microkernel::<EA>(
            &unit,
            &o.at(&unit),
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
