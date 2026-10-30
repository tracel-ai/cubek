use super::base::CubeQueryGradientMicrokernel;
use crate::attention::{G, Q, S};
use crate::attention_backward::query_gradient::plane::PlaneQueryGradientMicrokernel;
use crate::attention_backward::{ScoreWindows, drain_scaled};
use cubecl::tile::{Region, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

/// One cube's queries of some heads: each plane's dot products of the output with its gradient,
/// then its queries' gradient over the first `steps` steps of the cube's walk through `S`, each
/// plane taking every stage with the plane microkernel's step.
///
/// The cube stages its box of the query and of the output's gradient once, and the keys and values
/// a stage at a time for all its planes, holding as many stages as `ring` says, met on
/// `sync_cube`. Every tile is windowed to `cube`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_query_gradient_microkernel<EA: Float, ES: Float>(
    cube: &Region,
    query_gradient: &Tile,
    delta: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    bias: &Tile,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: CubeQueryGradientMicrokernel,
    #[comptime] ring: StageRing,
) {
    for plane in cube.planes() {
        let mut delta = delta.at(&plane);
        delta.dot::<EA>(&out_gradient.at(&plane), &out.at(&plane));
    }

    let walk = cube.walk().range(0, steps);
    let sum =
        PlaneQueryGradientMicrokernel::open_sum::<EA>(query_gradient, bias, key, microkernel.plane);
    // The query and the output's gradient span no key: staged once for every plane, before the
    // walk.
    let mut staged_query = query.stage_for(&walk, StageStorage::Strided, None);
    staged_query.copy_from(query);
    let mut staged_out_gradient = out_gradient.stage_for(&walk, StageStorage::Strided, None);
    staged_out_gradient.copy_from(out_gradient);
    let windows = ScoreWindows::open::<EA, ES>(&walk, comptime!(vec![G, Q, S]));

    // The staged operands and the dot products land before any plane reads them.
    sync_cube();

    let mut stages = Stages::smem(
        &walk,
        key,
        value,
        StageStorage::Strided,
        ring.stages_in_flight.get(),
        None,
    );

    stages.walk(walk, ring.prefetch, |slot, stage| {
        slot.consume(|key, value| {
            for plane in stage {
                PlaneQueryGradientMicrokernel::step::<EA>(
                    &plane,
                    &sum.at(&plane),
                    &windows.at(&plane),
                    &staged_query.at(&plane),
                    &staged_out_gradient.at(&plane),
                    &key.at(&plane),
                    &value.at(&plane),
                    &bias.at(&plane),
                    &logsumexp.at(&plane),
                    &delta.at(&plane),
                    scale,
                    microkernel.plane,
                    microkernel.score_levels(),
                    microkernel.score_gradient_levels(),
                );
            }
        });
    });

    for plane in cube.planes() {
        drain_scaled::<EA>(
            &sum.at(&plane),
            &query_gradient.at(&plane),
            scale,
            microkernel.plane.seq_q(),
        );
    }
}
