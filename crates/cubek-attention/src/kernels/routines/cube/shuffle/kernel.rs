use core::marker::PhantomData;

use super::base::CubeShuffleAttentionMicrokernel;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::output::cube::cube_unstaged_attention_microkernel;
use cubek_microkernel::attention::output::plane::PlaneShuffleAttention;
use cubek_microkernel::attention::{AttentionVisibility, B, TileLines, VisibleKeys};

/// Every cube running the cube microkernel on its box with planes of shuffles, each reading its keys
/// where they lie, over the keys its rows see.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_shuffle_attention_kernel<EA: Float, E: Float, N: Size>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeShuffleAttentionMicrokernel,
    #[comptime] value_dim: usize,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let q = query.tile(&partitioning);
    let k = TileLines::<E> {
        tile: key.tile(&partitioning),
        element: PhantomData,
    };
    let v = TileLines::<E> {
        tile: value.tile(&partitioning),
        element: PhantomData,
    };
    let o = out.tile(&partitioning);
    for cube in &partitioning {
        let visible = VisibleKeys::new(len, cube.origin(B), queries, visibility);
        let steps = visible.steps(
            &cube,
            comptime!(microkernel.seq_q()),
            comptime!(microkernel.region_keys()),
            1usize,
        );
        cube_unstaged_attention_microkernel::<
            EA,
            E,
            N,
            TileLines<E>,
            TileLines<E>,
            PlaneShuffleAttention<EA, N>,
        >(
            &cube,
            &o.at(&cube),
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
