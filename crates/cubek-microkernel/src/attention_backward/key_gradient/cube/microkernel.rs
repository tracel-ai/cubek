use super::base::CubeKeyGradientMicrokernel;
use crate::attention::{Q, S};
use crate::attention_backward::ScoreWindows;
use crate::attention_backward::key_gradient::plane::{KeyGradientSums, plane_key_gradient_step};
use crate::staging::StageRing;
use cubecl::tile::{Region, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's keys of one key head: each plane's keys' and values' gradients, over every head of
/// the group, each head walking its steps of queries from `first_step` on, each plane taking every
/// stage with the plane microkernel's step.
///
/// The cube stages its keys and values once, and each head's queries and output gradient a stage
/// at a time for all its planes, holding as many stages as `ring` says, met on `sync_cube`. Every
/// tile is windowed to `cube`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_staged_key_gradient_microkernel<EA: Float, ES: Float>(
    cube: &Region,
    key_gradient: &Tile,
    value_gradient: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    delta: &Tile,
    bias: &Tile,
    first_step: usize,
    scale: EA,
    #[comptime] microkernel: CubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
) {
    let heads = cube.walk();
    // Every head has the first's shape: the stages and windows are placed once, by its walk.
    let first_head = heads.region(0usize);
    let shape = first_head.walk();
    let windows = ScoreWindows::open::<EA, ES>(&shape, comptime!(vec![S, Q]));
    let sums = KeyGradientSums::open::<EA>(
        key_gradient,
        value_gradient,
        &windows.scores,
        &query.at(&first_head),
        &out_gradient.at(&first_head),
        comptime!(microkernel.plane),
    );
    // The keys and the values span no query: staged once for every plane, before the walk.
    let mut staged_key = key.stage_for(&shape, StageStorage::Strided, None);
    staged_key.copy_from(key);
    let mut staged_value = value.stage_for(&shape, StageStorage::Strided, None);
    staged_value.copy_from(value);
    sync_cube();
    // A runtime loop over the heads: one ring of stages, filled from each head's queries in turn.
    for g in 0..heads.total() {
        let head = heads.region(g);
        let queries = head.walk();
        let steps = queries.total();
        let mut stages = Stages::smem(
            &queries,
            &query.at(&head),
            &out_gradient.at(&head),
            StageStorage::Strided,
            comptime!(ring.stages_in_flight.get()),
            None,
        );
        stages.walk(
            queries.range(first_step, steps),
            comptime!(ring.prefetch),
            |slot, stage| {
                slot.consume(|query, out_gradient| {
                    for plane in stage {
                        plane_key_gradient_step::<EA>(
                            &plane,
                            &sums.at(&plane),
                            &windows.at(&plane),
                            &staged_key.at(&plane),
                            &staged_value.at(&plane),
                            &query.at(&plane),
                            &out_gradient.at(&plane),
                            &bias.at(&plane),
                            &logsumexp.at(&plane),
                            &delta.at(&plane),
                            scale,
                            comptime!(microkernel.plane),
                            comptime!(microkernel.score_levels()),
                            comptime!(microkernel.score_gradient_levels()),
                        );
                    }
                });
            },
        );
    }
    for plane in cube.planes() {
        sums.at(&plane).store::<EA>(
            &key_gradient.at(&plane),
            &value_gradient.at(&plane),
            scale,
            comptime!(microkernel.plane),
        );
    }
}
