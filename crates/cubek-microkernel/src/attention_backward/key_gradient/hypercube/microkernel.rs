use super::base::HypercubeKeyGradientMicrokernel;
use crate::attention::{DV, Q, S};
use crate::attention_backward::key_gradient::plane::{KeyGradientSums, plane_key_gradient_step};
use crate::attention_backward::{ScoreWindows, drain_scaled};
use crate::matmul::plane::plane_fragment_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Region, Semiring, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

/// One cube's keys of one key head, as the cube microkernel walks them, every plane also adding
/// its share of each stage's query gradient into `query_gradient`, a sum every cube that sees
/// those queries adds into: `scale · score_gradient · key`, out of the score's gradient the
/// plane step leaves cast in its window. `query_gradient` arrives at zero and is only added into.
///
/// Every tile is windowed to `cube`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_key_gradient_microkernel<EA: Float, ES: Float>(
    cube: &Region,
    key_gradient: &Tile,
    value_gradient: &Tile,
    query_gradient: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    delta: &Tile,
    bias: &Tile,
    first_step: usize,
    scale: EA,
    #[comptime] microkernel: HypercubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
) {
    let plane_microkernel = comptime!(microkernel.cube.plane);
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
        plane_microkernel,
    );
    // The keys and the values span no query: staged once for every plane, before the walk.
    let mut staged_key = key.stage_for(&shape, StageStorage::Strided, None);
    staged_key.copy_from(key);
    let mut staged_value = value.stage_for(&shape, StageStorage::Strided, None);
    staged_value.copy_from(value);
    sync_cube();
    let wide = ComptimeOption::<Tile>::new_None();
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
                        let windows = windows.at(&plane);
                        let key = staged_key.at(&plane);
                        plane_key_gradient_step::<EA>(
                            &plane,
                            &sums.at(&plane),
                            &windows,
                            &key,
                            &staged_value.at(&plane),
                            &query.at(&plane),
                            &out_gradient.at(&plane),
                            &bias.at(&plane),
                            &logsumexp.at(&plane),
                            &delta.at(&plane),
                            scale,
                            plane_microkernel,
                            comptime!(microkernel.score_levels()),
                            comptime!(microkernel.score_gradient_levels()),
                        );
                        // The step leaves the score's gradient cast in its window: its product
                        // with the keys is this plane's share of the stage's query gradient.
                        let query_gradient = query_gradient.at(&plane);
                        let mut sum = query_gradient
                            .accumulator::<EA>(
                                &windows.cast,
                                &key,
                                comptime!(microkernel.query_gradient.tile_instruction()),
                                Semiring::SUM_PROD,
                            )
                            .with_scratch(comptime!(plane_microkernel.settings.scratch));
                        sum.zero();
                        plane_fragment_microkernel(
                            &plane,
                            &sum.clone(),
                            &wide,
                            &windows.cast.clone(),
                            &key,
                            comptime!(microkernel.query_gradient.tile_instruction()),
                            comptime!(Some(DV)),
                            comptime!(MatmulOperands::all(MatmulRead::AsBound)),
                        );
                        drain_scaled::<EA>(
                            &sum,
                            &query_gradient,
                            scale,
                            comptime!(microkernel.seq_q()),
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
            plane_microkernel,
        );
    }
}
