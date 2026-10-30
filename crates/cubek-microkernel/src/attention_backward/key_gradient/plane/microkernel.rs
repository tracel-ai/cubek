use super::base::PlaneKeyGradientMicrokernel;
use crate::attention::{D, DV, Q, S};
use crate::attention_backward::{ScoreWindows, drain_scaled};
use crate::matmul::MatmulOperands;
use crate::matmul::plane::plane_fragment_microkernel;
use cubecl::tile::{Level, Region, Semiring, Space, StageStorage, Stages, Tile};
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

    // The windows span the keys and one head's queries, the axes of every product out of them;
    // a score held in fragments needs none.
    let windows = ScoreWindows::held_or_open::<EA, ES>(&shape, microkernel.holder());
    let sums = KeyGradientSums::open::<EA>(
        key_gradient,
        value_gradient,
        &ScoreWindows::scores_box::<EA>(&windows, &shape, key, microkernel.score),
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
                    PlaneKeyGradientMicrokernel::step::<EA, ES>(
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
            .with_scratch(microkernel.sum_scratch());
        key.zero();
        let mut value = value_gradient
            .accumulator::<EA>(
                scores,
                out_gradient,
                microkernel.value_gradient.tile_instruction(),
                Semiring::SUM_PROD,
            )
            .with_scratch(microkernel.sum_scratch());
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
    /// into the values' sum, then the gradient against them from the output's gradient, the
    /// gradient against the scores from `delta`, and its product with the queries added into the
    /// keys' sum.
    ///
    /// `stage` is the plane's region of the stage; every tile is windowed to the plane. The levels
    /// are each window's own grid under the walker's planes, which a window is placed in.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step<EA: Float, ES: Float>(
        stage: &Region,
        sums: &KeyGradientSums,
        windows: &ComptimeOption<ScoreWindows>,
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
        #[comptime]
        match windows {
            ComptimeOption::Some(windows) => Self::window_step::<EA>(
                stage,
                sums,
                windows,
                key,
                value,
                query,
                out_gradient,
                bias,
                logsumexp,
                delta,
                scale,
                microkernel,
                score_levels,
                score_gradient_levels,
            ),
            ComptimeOption::None => Self::held_step::<EA, ES>(
                sums,
                key,
                value,
                query,
                out_gradient,
                bias,
                logsumexp,
                delta,
                scale,
                microkernel,
                score_levels,
                score_gradient_levels,
            ),
        }
    }

    /// [`step`](Self::step) with the score and the gradient against its probabilities in the
    /// plane's windows, met on `sync_plane` around each.
    #[allow(clippy::too_many_arguments)]
    fn window_step<EA: Float>(
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

    /// [`step`](Self::step) with the score and the gradient against its probabilities held in
    /// their own manual-mma fragments: the probabilities and the gradient against the score taken
    /// in place, each cast in its registers as the `A` of the gradient that contracts it, a run of
    /// two along the stage's queries. Nothing passes through shared memory, so the plane meets on
    /// nothing.
    #[allow(clippy::too_many_arguments)]
    fn held_step<EA: Float, ES: Float>(
        sums: &KeyGradientSums,
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
        let space = comptime!(Space::new(&[
            (S, microkernel.seq_kv()),
            (Q, microkernel.seq_q())
        ]));
        let scores = ScoreWindows::contract_held::<EA>(
            key,
            query,
            comptime!(space.clone()),
            microkernel.score,
            score_levels,
        );
        scores.along(Q).probabilities::<EA>(scale, bias, logsumexp);
        ScoreWindows::contract_fragments(
            &sums.value,
            &scores.cast::<ES>(),
            out_gradient,
            microkernel.value_gradient,
        );

        let gradients = ScoreWindows::contract_held::<EA>(
            value,
            out_gradient,
            space,
            microkernel.score_gradient,
            score_gradient_levels,
        );
        gradients
            .along(Q)
            .softmax_gradient::<EA>(&scores.along(Q), delta);
        ScoreWindows::contract_fragments(
            &sums.key,
            &gradients.cast::<ES>(),
            query,
            microkernel.key_gradient,
        );
    }
}
