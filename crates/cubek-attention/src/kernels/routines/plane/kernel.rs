use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, plane_fragment_attention_microkernel,
};
use cubek_microkernel::attention::{AttentionVisibility, B, CubePlaneMicrokernel, S, VisibleKeys};
use cubek_microkernel::staging::StageRing;

/// Every plane of the launch owning its rows, each running them through stages of its own
/// ([`plane_fragment_attention_microkernel`]) over the keys its rows see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_attention_kernel<EA: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    for plane in &partitioning {
        let visible = VisibleKeys::new(len, plane.origin(B), queries, visibility);
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(
            &plane,
            comptime!(microkernel.seq_q()),
            comptime!(microkernel.seq_kv()),
            1usize,
        );
        plane_fragment_attention_microkernel::<EA, ES>(
            &plane,
            &o.at(&plane),
            &q.at(&plane),
            &k.at(&plane),
            &v.at(&plane),
            &bias.at(&plane),
            steps,
            EA::cast_from(scale),
            microkernel,
            ring,
            planes,
        );
    }
}
