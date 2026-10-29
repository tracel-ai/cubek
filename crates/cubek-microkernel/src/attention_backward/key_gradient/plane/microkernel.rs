use super::base::PlaneKeyGradientMicrokernel;
use crate::attention::{D, DV, Q, S};
use crate::attention_backward::{ScoreWindows, drain_scaled};
use crate::matmul::MatmulOperands;
use crate::matmul::plane::plane_fragment_microkernel;
use cubecl::tile::{Level, Region, Semiring, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

/// One plane's keys of one key head: `value_gradient = Σ_{g, q} probabilities · out_gradient` and
/// `key_gradient = scale · Σ_{g, q} probabilities ⊙ (value · out_gradientᵀ − delta) · query`, the
/// scores held keys first, summed in `EA`, the probabilities read back from each query's
/// `logsumexp`. Each head of the group walks its steps of queries from `first_step` on: the
/// queries before it see none of the plane's keys.
///
/// The plane stages its keys and values once, as its own copies, and each head's queries and
/// output gradient a stage at a time, holding as many stages as `ring` says. Its
/// scores, its gradient against the probabilities and either cast to `ES` lie in windows of its
/// own, met on `sync_plane`. Every tile is windowed to `keys`; `planes` is how many planes along
/// the keys share the cube, which its own grids are placed under.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_key_gradient_microkernel<EA: Float, ES: Float>(
    keys: &Region,
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
    #[comptime] microkernel: PlaneKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
) {
    let heads = keys.walk();

    // Every head has the first's shape: the stages and windows are placed once, by its walk.
    let first_head = heads.region(0usize);
    let shape = first_head.walk();

    // The windows span the keys and one head's queries, the axes of every product out of them.
    let windows = ScoreWindows::open::<EA, ES>(&shape, comptime!(vec![S, Q]));
    let sums = KeyGradientSums::open::<EA>(
        key_gradient,
        value_gradient,
        &windows.scores,
        &query.at(&first_head),
        &out_gradient.at(&first_head),
        microkernel,
    );

    // The keys and the values span no query: staged once, before the walk.
    let mut staged_key = key.stage_for(&shape, StageStorage::Strided, None);
    staged_key.copy_from(key);
    let mut staged_value = value.stage_for(&shape, StageStorage::Strided, None);
    staged_value.copy_from(value);
    sync_plane();

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
                let (bias, logsumexp, delta) =
                    (bias.at(stage), logsumexp.at(stage), delta.at(stage));
                slot.consume(|query, out_gradient| {
                    PlaneKeyGradientMicrokernel::step::<EA>(
                        stage,
                        &sums,
                        &windows,
                        &staged_key,
                        &staged_value,
                        query,
                        out_gradient,
                        &bias,
                        &logsumexp,
                        &delta,
                        scale,
                        microkernel,
                        microkernel.score_levels(&[(S, planes)]),
                        microkernel.score_gradient_levels(&[(S, planes)]),
                    );
                });
            },
        );
    }

    sums.store::<EA>(key_gradient, value_gradient, scale, microkernel);
}

/// The sums every stage's gradients add into, at `EA`, held across every stage of every head:
/// the keys' and the values'.
#[derive(CubeType, Clone)]
pub(crate) struct KeyGradientSums {
    pub key: Tile,
    pub value: Tile,
}

#[cube]
impl KeyGradientSums {
    /// Both opened at zero on the gradients, against a step's `scores` and the query and output
    /// gradient of one head, each with the scratch its scale bounces through.
    pub fn open<EA: Float>(
        key_gradient: &Tile,
        value_gradient: &Tile,
        scores: &Tile,
        query: &Tile,
        out_gradient: &Tile,
        #[comptime] microkernel: PlaneKeyGradientMicrokernel,
    ) -> KeyGradientSums {
        let mut key = key_gradient
            .accumulator::<EA>(
                scores,
                query,
                microkernel.key_gradient.tile_instruction(),
                Semiring::SUM_PROD,
            )
            .with_scratch(microkernel.settings.scratch);
        key.zero();
        let mut value = value_gradient
            .accumulator::<EA>(
                scores,
                out_gradient,
                microkernel.value_gradient.tile_instruction(),
                Semiring::SUM_PROD,
            )
            .with_scratch(microkernel.settings.scratch);
        value.zero();
        KeyGradientSums { key, value }
    }

    /// `plane`'s sums.
    pub fn at(&self, plane: &Region) -> KeyGradientSums {
        KeyGradientSums {
            key: self.key.at(plane),
            value: self.value.at(plane),
        }
    }

    /// The keys' sum scaled, and both drained into the gradients, all windowed to one plane's
    /// keys.
    pub fn store<EA: Float>(
        &self,
        key_gradient: &Tile,
        value_gradient: &Tile,
        scale: EA,
        #[comptime] microkernel: PlaneKeyGradientMicrokernel,
    ) {
        drain_scaled::<EA>(&self.key, key_gradient, scale, microkernel.seq_kv());
        self.value.drained_into(value_gradient);
    }
}

#[cube]
impl PlaneKeyGradientMicrokernel {
    /// One stage of one plane's keys: their scores against the stage's queries, keys first, the
    /// probabilities read back from `logsumexp` and their product with the output's gradient added
    /// into the values' sum, then the gradient against them from the output's gradient, the gradient
    /// against the scores from `delta`, and its product with the queries added into the keys' sum.
    ///
    /// `stage` is the plane's region of the stage; every tile is windowed to the plane. The levels
    /// are each window's own grid under the walker's planes, which a window is placed in.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step<EA: Float>(
        stage: &Region,
        sums: &KeyGradientSums,
        windows: &ScoreWindows,
        key: &Tile,
        value: &Tile,
        query: &Tile,
        out_gradient: &Tile,
        bias: &Tile,
        logsumexp: &Tile,
        delta: &Tile,
        scale: EA,
        #[comptime] microkernel: Self,
        #[comptime] score_levels: Vec<Level>,
        #[comptime] score_gradient_levels: Vec<Level>,
    ) {
        let (scores, gradients) = (&windows.scores, &windows.gradients);
        let mut cast = windows.cast.clone();
        let wide = ComptimeOption::<Tile>::new_None();
        ScoreWindows::contract_into::<EA>(scores, key, query, microkernel.score, score_levels);

        // The scores' cells were stored in the fragments' layout; the softmax reads them along the
        // queries.
        sync_plane();

        scores.along(Q).probabilities::<EA>(scale, bias, logsumexp);
        sync_plane();

        cast.copy_cast_from(scores);

        // The probabilities land before the fragments load them.
        sync_plane();

        // A tile's clone is a handle to the same cells: the shared microkernel reads the
        // probabilities and adds into the sum through it.
        plane_fragment_microkernel(
            stage,
            &sums.value.clone(),
            &wide,
            &cast.clone(),
            out_gradient,
            microkernel.value_gradient.tile_instruction(),
            comptime!(Some(D)),
            MatmulOperands::AS_BOUND,
        );

        ScoreWindows::contract_into::<EA>(
            gradients,
            value,
            out_gradient,
            microkernel.score_gradient,
            score_gradient_levels,
        );

        // The gradient against the probabilities lands, and the cast probabilities are read, before
        // either window is written again.
        sync_plane();

        gradients
            .along(Q)
            .softmax_gradient::<EA>(&scores.along(Q), delta);
        sync_plane();

        cast.copy_cast_from(gradients);

        // The gradient lands before the fragments load it.
        sync_plane();

        plane_fragment_microkernel(
            stage,
            &sums.key.clone(),
            &wide,
            &cast.clone(),
            query,
            microkernel.key_gradient.tile_instruction(),
            comptime!(Some(DV)),
            MatmulOperands::AS_BOUND,
        );
    }
}
