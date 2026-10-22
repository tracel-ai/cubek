use cubecl::tile::{Partitioning, StageStorage, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, cube_staged_attention_microkernel,
};
use cubek_microkernel::attention::{
    AttentionVisibility, B, CubePlaneMicrokernel, CubeStagedPlane, S, VisibleKeys,
};

/// Every cube running the cube microkernel on its box with planes that read its stage, held as
/// `storage`, over the keys its rows see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_attention_kernel<
    EA: Float,
    E: Float,
    ES: Float,
    M: CubePlaneMicrokernel + CubeStagedPlane,
>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    len: &Tensor<u32>,
    pages: ComptimeOption<&Tensor<u32>>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeAttentionMicrokernel<M>,
    #[comptime] storage: StageStorage,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E, ES)] _elements: [ElemType; 3],
) {
    let q = query.tile(&partitioning);
    let key = key.tile_paged(&partitioning, pages);
    let value = value.tile_paged(&partitioning, pages);
    let o = out.tile(&partitioning);
    for cube in &partitioning {
        let visible = VisibleKeys::new(len, cube.origin(B), queries, visibility);
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(
            &cube,
            comptime!(microkernel.seq_q()),
            comptime!(microkernel.seq_kv()),
            1usize,
        );
        cube_staged_attention_microkernel::<EA, E, M::Attention<EA, E, ES>>(
            &cube,
            &o.at(&cube),
            &q.at(&cube),
            &k.at(&cube),
            &v.at(&cube),
            &bias.at(&cube),
            steps,
            EA::cast_from(scale),
            microkernel,
            comptime!(storage.clone()),
        );
    }
}
