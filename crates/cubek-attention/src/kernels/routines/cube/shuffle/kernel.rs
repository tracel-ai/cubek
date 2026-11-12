use core::marker::PhantomData;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, cube_unstaged_attention_microkernel,
};
use cubek_microkernel::attention::plane::TileStepReads;
use cubek_microkernel::attention::{AttentionVisibility, VisibleKeys};

/// Every cube running the cube microkernel on its box with planes of shuffles, each reading its
/// keys where they lie, over the keys its rows see, and writing each row's logsumexp where the
/// launch binds one. Each operand is read in its own element and widened into the sum, `EA`:
/// the query in `EQ`, the keys in `EK`, the values in `EV`; the output is written in `EO`.
#[cube(launch_unchecked)]
pub fn cube_shuffle_attention_kernel<
    EA: Float,
    EQ: Float,
    EK: Float,
    EV: Float,
    EO: Float,
    N: Size,
>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &TileTensor<'_>,
    logsumexp: ComptimeOption<&TileTensor<'static>>,
    mask: ComptimeOption<&TileTensor<'static>>,
    len: &Tensor<u32>,
    query_ends: ComptimeOption<&Tensor<u32>>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] value_dim: usize,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, EQ, EK, EV, EO)] _elements: [ElemType; 5],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));

    let q = query.tile(&partitioning);
    let k = TileStepReads::<EK> {
        tile: key.tile(&partitioning),
        element: PhantomData,
    };
    let v = TileStepReads::<EV> {
        tile: value.tile(&partitioning),
        element: PhantomData,
    };

    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));

    for sequence in &partitioning {
        for cube in sequence {
            let visible = VisibleKeys::new(len, &query_ends, &cube, queries, &mask, visibility);
            let steps = visible.steps(
                &cube,
                microkernel.seq_q(),
                microkernel.region_keys(),
                1usize,
            );

            cube_unstaged_attention_microkernel::<
                EA,
                EQ,
                EO,
                N,
                TileStepReads<EK>,
                TileStepReads<EV>,
            >(
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
}
