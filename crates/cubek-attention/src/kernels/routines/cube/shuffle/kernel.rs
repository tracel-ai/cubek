use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl::tile::{Partitioning, Tile, TileArg};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, cube_unstaged_attention_microkernel,
};
use cubek_microkernel::attention::plane::TileStepReads;
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};

/// Every cube running the cube microkernel on its box with planes of shuffles, each reading its
/// keys where they lie, over the keys its rows see, and writing each row's logsumexp where the
/// launch binds one.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_shuffle_attention_kernel<EA: Float, E: Float, N: Size>(
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
    #[comptime] value_dim: usize,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let k = TileStepReads::<E> {
        tile: key.tile(&partitioning),
        element: PhantomData,
    };
    let v = TileStepReads::<E> {
        tile: value.tile(&partitioning),
        element: PhantomData,
    };
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));
    for cube in &partitioning {
        let visible = VisibleKeys::new(len, &cube, queries, &mask, visibility);
        let steps = visible.steps(
            &cube,
            microkernel.seq_q(),
            microkernel.region_keys(),
            1usize,
        );

        cube_unstaged_attention_microkernel::<EA, E, N, TileStepReads<E>, TileStepReads<E>>(
            &cube,
            &o.at(&cube),
            &lse.clone().map(|lse| lse.at(&cube)),
            &q.at(&cube),
            &k,
            &v,
            &visible,
            steps,
            EA::cast_from(scale),
            value_dim,
            microkernel,
        );
    }
}
