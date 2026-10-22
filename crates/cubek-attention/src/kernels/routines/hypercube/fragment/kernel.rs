use super::base::HypercubeFragmentAttentionMicrokernel;
use cubecl::tile::{Partitioning, StageStorage, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::cube::cube_staged_attention_partial_microkernel;
use cubek_microkernel::attention::hypercube::HypercubeMergeShape;
use cubek_microkernel::attention::plane::PlaneFragmentAttention;
use cubek_microkernel::attention::{AttentionVisibility, B, S, VisibleKeys};

/// Every cube walking its run of its box with plane fragments that read its stage, over the keys
/// its queries see, each plane leaving its partial in its slot
/// ([`cube_staged_attention_partial_microkernel()`]), for a second launch to merge. The output is
/// read for the shape of what its planes sum into.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_fragment_attention_kernel<EA: Float, E: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    slot_out: &mut Tensor<Vector<EA, Const<1>>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    len: &Tensor<u32>,
    pages: ComptimeOption<&Tensor<u32>>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeFragmentAttentionMicrokernel,
    #[comptime] merge: HypercubeMergeShape,
    #[comptime] visibility: AttentionVisibility,
    #[comptime] storage: StageStorage,
    #[define(EA, E, ES)] _elements: [ElemType; 3],
) {
    // The cube level's walk, this cube's run of the steps of keys: once for the staging ring,
    // once for where the slots lie.
    let run = partitioning.walk();
    let first_step = run.region(0);
    let visible = VisibleKeys::new(len, first_step.origin(B), queries, visibility);
    let q = query.tile(&partitioning);
    let k = key
        .tile_paged(&partitioning, pages)
        .within(S, 0, visible.attended as usize);
    let v = value
        .tile_paged(&partitioning, pages)
        .within(S, 0, visible.attended as usize);
    let o = out.tile(&partitioning);
    let bias = visible.bias::<EA>(&partitioning);
    let steps = visible.steps(
        &first_step,
        comptime!(microkernel.cube.queries()),
        comptime!(microkernel.cube.keys()),
        comptime!(merge.cubes),
    );
    // The query and the output span no key: their box is the first step's.
    let _ = cube_staged_attention_partial_microkernel::<
        EA,
        E,
        Const<1>,
        PlaneFragmentAttention<EA, ES>,
    >(
        partitioning.walk(),
        &run,
        &o.at(&first_step),
        &q.at(&first_step),
        &k,
        &v,
        &bias,
        steps,
        EA::cast_from(scale),
        slot_out,
        slot_max,
        slot_sum,
        comptime!(microkernel.cube),
        comptime!(merge.slots.splits),
        storage,
    );
}
