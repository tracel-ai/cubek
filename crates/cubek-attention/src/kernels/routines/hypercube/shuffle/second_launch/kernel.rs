use core::marker::PhantomData;

use crate::kernels::HypercubeShuffleAttentionMicrokernel;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::cube_unstaged_attention_partial_microkernel;
use cubek_microkernel::attention::hypercube::HypercubeMergeShape;
use cubek_microkernel::attention::packed_cache::PackedCacheArg;
use cubek_microkernel::attention::plane::PlaneShuffleAttention;
use cubek_microkernel::attention::{AttentionVisibility, B, PackedLines, TileLines, VisibleKeys};

/// Every cube walking its run of its box over the keys its queries see, its planes reading the
/// keys and values from tiles, leaving its partial in its slot for a second launch to merge.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_attention_partial_kernel<EA: Float, E: Float, N: Size>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    len: &Tensor<u32>,
    pages: ComptimeOption<&Tensor<u32>>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeShuffleAttentionMicrokernel,
    #[comptime] merge: HypercubeMergeShape,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let key = TileLines::<E> {
        tile: key.tile_paged(&partitioning, pages),
        element: PhantomData,
    };
    let value = TileLines::<E> {
        tile: value.tile_paged(&partitioning, pages),
        element: PhantomData,
    };
    // The cube level's walk: this cube's run of the steps, or pages, of keys.
    let run = partitioning.walk();
    let first_step = run.region(0);
    let q = query.tile(&partitioning);
    let visible = VisibleKeys::new(len, first_step.origin(B), queries, visibility);
    let steps = visible.steps(
        &first_step,
        comptime!(microkernel.cube.seq_q()),
        comptime!(microkernel.cube.region_keys()),
        comptime!(merge.cubes),
    );
    let _ = cube_unstaged_attention_partial_microkernel::<
        EA,
        E,
        N,
        TileLines<E>,
        TileLines<E>,
        PlaneShuffleAttention<EA, N>,
    >(
        run,
        // The query spans no key: its box is the first step's.
        &q.at(&first_step),
        &key,
        &value,
        &visible,
        steps,
        EA::cast_from(scale),
        slot_out,
        slot_max,
        slot_sum,
        comptime!(microkernel.cube),
        comptime!(merge.slots.splits),
        comptime!(merge.slots.val_dim),
    );
}

/// [`hypercube_attention_partial_kernel`] over a packed cache, its planes decoding the keys and
/// values as they read them ([`PackedLines`]).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_attention_partial_packed_kernel<EA: Float, E: Float, N: Size>(
    query: &TileArg<'_>,
    cache: &PackedCacheArg,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeShuffleAttentionMicrokernel,
    #[comptime] merge: HypercubeMergeShape,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let key = PackedLines::keys(cache);
    let value = PackedLines::values(cache);
    // The cube level's walk: this cube's run of the steps, or pages, of keys.
    let run = partitioning.walk();
    let first_step = run.region(0);
    let q = query.tile(&partitioning);
    let visible = VisibleKeys::new(len, first_step.origin(B), queries, visibility);
    let steps = visible.steps(
        &first_step,
        comptime!(microkernel.cube.seq_q()),
        comptime!(microkernel.cube.region_keys()),
        comptime!(merge.cubes),
    );
    let _ = cube_unstaged_attention_partial_microkernel::<
        EA,
        E,
        N,
        PackedLines,
        PackedLines,
        PlaneShuffleAttention<EA, N>,
    >(
        run,
        // The query spans no key: its box is the first step's.
        &q.at(&first_step),
        &key,
        &value,
        &visible,
        steps,
        EA::cast_from(scale),
        slot_out,
        slot_max,
        slot_sum,
        comptime!(microkernel.cube),
        comptime!(merge.slots.splits),
        comptime!(merge.slots.val_dim),
    );
}
