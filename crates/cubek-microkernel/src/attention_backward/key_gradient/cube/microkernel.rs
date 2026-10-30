use super::base::CubeKeyGradientMicrokernel;
use super::share::{QueryGradientShare, StageQueryGradientShare};
use crate::attention::{Q, S};
use crate::attention_backward::ScoreWindows;
use crate::attention_backward::key_gradient::plane::{
    KeyGradientSums, PlaneKeyGradientMicrokernel,
};
use cubecl::tile::{Region, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

/// One cube's keys of one key head: each plane's keys' and values' gradients, over every head of
/// the group, each head walking its steps of queries from `first_step` on, each plane taking every
/// stage with the plane microkernel's step. Where a `query_gradient` share is given, each plane
/// then adds its part of each stage's query gradient, the planes meeting in shared memory so the
/// cube adds into the sum once a stage ([`QueryGradientShare`]).
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
    query_gradient: &ComptimeOption<QueryGradientShare>,
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
    let plane_microkernel = microkernel.plane;
    let heads = cube.walk();

    // Every head has the first's shape: the stages and windows are placed once, by its walk.
    let first_head = heads.region(0usize);
    let shape = first_head.walk();
    let windows = ScoreWindows::held_or_open::<EA, ES>(
        &shape,
        comptime!(vec![S, Q]),
        plane_microkernel.holder(),
    );
    #[comptime]
    let left = match query_gradient {
        ComptimeOption::Some(_) => {
            ScoreWindows::share_window::<ES>(&shape, plane_microkernel.holder())
        }
        ComptimeOption::None => ComptimeOption::<Tile>::new_None(),
    };
    let sums = KeyGradientSums::open::<EA>(
        key_gradient,
        value_gradient,
        &ScoreWindows::scores_box::<EA>(&windows, &shape, key, plane_microkernel.score),
        &query.at(&first_head),
        &out_gradient.at(&first_head),
        plane_microkernel,
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
            ring.stages_in_flight.get(),
            None,
        );

        stages.walk(
            queries.range(first_step, steps),
            ring.prefetch,
            |slot, stage| {
                slot.consume(|query, out_gradient| {
                    #[comptime]
                    let mut share = match query_gradient {
                        ComptimeOption::Some(share) => {
                            ComptimeOption::new_Some(share.open::<EA>(stage))
                        }
                        ComptimeOption::None => {
                            ComptimeOption::<StageQueryGradientShare<EA>>::new_None()
                        }
                    };

                    for plane in stage {
                        let windows = ScoreWindows::held_at(&windows, &plane);
                        let left = ScoreWindows::window_at(&left, &plane);
                        let key = staged_key.at(&plane);
                        PlaneKeyGradientMicrokernel::step::<EA, ES>(
                            &plane,
                            &sums.at(&plane),
                            &windows,
                            &left,
                            &key,
                            &staged_value.at(&plane),
                            &query.at(&plane),
                            &out_gradient.at(&plane),
                            &bias.at(&plane),
                            &logsumexp.at(&plane),
                            &delta.at(&plane),
                            scale,
                            plane_microkernel,
                            microkernel.score_levels(),
                            microkernel.score_gradient_levels(),
                        );

                        // The step leaves the score's gradient cast in its window: its product
                        // with the keys is this plane's part of the stage's query gradient.
                        #[comptime]
                        match &share {
                            ComptimeOption::Some(share) => share.add(
                                &plane,
                                &ScoreWindows::cast_of(&windows, &left),
                                &key,
                                scale,
                                plane_microkernel.settings.scratch,
                                microkernel.seq_q(),
                            ),
                            ComptimeOption::None => {}
                        }
                    }

                    #[comptime]
                    match share {
                        ComptimeOption::Some(ref mut share) => share.write(),
                        ComptimeOption::None => {}
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
            plane_microkernel,
        );
    }
}
