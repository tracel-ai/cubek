use super::base::PlaneQueryGradientMicrokernel;
use crate::attention::{D, G, Q, S};
use crate::attention_backward::{ScoreWindows, contract_into_window};
use crate::matmul::dense::plane::plane_fragment_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use crate::staging::StageRing;
use cubecl::tile::{Level, Region, Semiring, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's queries of one head: `delta = Σ_dv out_gradient · out`, then
/// `query_gradient = scale · Σ_s probabilities ⊙ (out_gradient · valueᵀ − delta) · key` over the
/// first `steps` steps of the queries' walk through `S`, summed in `EA`, the probabilities read
/// back from each query's `logsumexp`.
///
/// The plane stages its query and the output's gradient once, as its own copies, and its keys and
/// values a stage at a time, holding as many stages as `ring` says. Its scores, its gradient
/// against the probabilities and that gradient cast to `ES` lie in windows of its own, met on
/// `sync_plane`. Every tile is windowed to `queries`; `planes` is how many planes along the
/// queries share the cube, which its own grids are placed under. The dot product is written
/// before the walk, for the keys' walk to read.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_query_gradient_microkernel<EA: Float, ES: Float>(
    queries: &Region,
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
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
) {
    // A tile's clone is a handle to the same cells: the dot product lands in `delta`.
    let mut delta = delta.clone();
    delta.dot::<EA>(out_gradient, out);
    let walk = queries.walk().range(0, steps);
    let sum = query_gradient_sum::<EA>(query_gradient, bias, key, microkernel);
    // The query and the output's gradient span no key: staged once, before the walk.
    let mut staged_query = query.stage_for(&walk, StageStorage::Strided, None);
    staged_query.copy_from(query);
    let mut staged_out_gradient = out_gradient.stage_for(&walk, StageStorage::Strided, None);
    staged_out_gradient.copy_from(out_gradient);
    let windows = ScoreWindows::open::<EA, ES>(&walk, comptime!(vec![G, Q, S]));
    // The query, the output's gradient and the dot products land before any is read.
    sync_plane();
    let mut stages = Stages::smem(
        &walk,
        key,
        value,
        StageStorage::Strided,
        comptime!(ring.stages_in_flight.get()),
        None,
    );
    stages.walk(walk, comptime!(ring.prefetch), |slot, stage| {
        let bias = bias.at(stage);
        slot.consume(|key, value| {
            plane_query_gradient_step::<EA>(
                stage,
                &sum,
                &windows,
                &staged_query,
                &staged_out_gradient,
                key,
                value,
                &bias,
                logsumexp,
                &delta,
                scale,
                microkernel,
                comptime!(microkernel.score_levels(&[(Q, planes)])),
                comptime!(microkernel.score_gradient_levels(&[(Q, planes)])),
            );
        });
    });
    store_query_gradient::<EA>(&sum, query_gradient, scale, microkernel);
}

/// The sum every stage's query gradient adds into, at `EA`, opened at zero on `query_gradient`
/// against the keys, with the scratch its scale bounces through.
#[cube]
pub(crate) fn query_gradient_sum<EA: Float>(
    query_gradient: &Tile,
    bias: &Tile,
    key: &Tile,
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
) -> Tile {
    let mut sum = query_gradient
        .accumulator::<EA>(
            bias,
            key,
            comptime!(microkernel.query_gradient.tile_instruction()),
            Semiring::SUM_PROD,
        )
        .with_scratch(comptime!(microkernel.settings.scratch));
    sum.zero();
    sum
}

/// One stage of one plane's queries: their scores against the stage's keys, the probabilities
/// read back from `logsumexp`, the gradient against them from the stage's values, the gradient
/// against the scores from `delta`, and its product with the keys added into `sum`.
///
/// `stage` is the plane's region of the stage; every tile is windowed to the plane. The levels
/// are each window's own grid under the walker's planes, which a window is placed in.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plane_query_gradient_step<EA: Float>(
    stage: &Region,
    sum: &Tile,
    windows: &ScoreWindows,
    query: &Tile,
    out_gradient: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    logsumexp: &Tile,
    delta: &Tile,
    scale: EA,
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
    #[comptime] score_levels: Vec<Level>,
    #[comptime] score_gradient_levels: Vec<Level>,
) {
    let (scores, gradients) = (&windows.scores, &windows.gradients);
    let wide = ComptimeOption::<Tile>::new_None();
    contract_into_window::<EA>(
        scores,
        query,
        key,
        comptime!(microkernel.score),
        score_levels,
    );
    // The scores' cells were stored in the fragments' layout; the softmax reads them by rows.
    sync_plane();
    scores.along(S).probabilities::<EA>(scale, bias, logsumexp);
    contract_into_window::<EA>(
        gradients,
        out_gradient,
        value,
        comptime!(microkernel.score_gradient),
        score_gradient_levels,
    );
    // The probabilities and the gradient against them land before both are read.
    sync_plane();
    gradients
        .along(S)
        .softmax_gradient::<EA>(&scores.along(S), delta);
    sync_plane();
    let mut cast = windows.cast.clone();
    cast.copy_cast_from(gradients);
    // The gradient lands before the fragments load it.
    sync_plane();
    // A tile's clone is a handle to the same cells: the shared microkernel reads the gradient and
    // adds into the sum through it.
    plane_fragment_microkernel(
        stage,
        &sum.clone(),
        &wide,
        &cast,
        key,
        comptime!(microkernel.query_gradient.tile_instruction()),
        comptime!(None),
        comptime!(MatmulOperands::all(MatmulRead::AsBound)),
    );
}

/// `sum` scaled and drained into `query_gradient`, both windowed to one plane's queries.
#[cube]
pub(crate) fn store_query_gradient<EA: Float>(
    sum: &Tile,
    query_gradient: &Tile,
    scale: EA,
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
) {
    let mut scales = Array::<EA>::new(comptime!(microkernel.seq_q()));
    #[unroll]
    for q in 0..comptime!(microkernel.seq_q()) {
        scales[q] = scale;
    }
    sum.along(D).mul(&scales);
    sum.drained_into(query_gradient);
}
