use super::base::{PlaneFragmentAttentionMicrokernel, ScoreHolder};
use crate::attention::{
    CubePlaneAttention, CubePlaneAttentionExpand, CubePlaneLayout, CubePlaneMicrokernel, D, DV, G,
    Q, S, StagedPlaneOperands,
};
use crate::matmul::dense::plane::plane_fragment_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use crate::staging::StageRing;
use core::marker::PhantomData;

use cubecl::tile::{
    Level, Region, Semiring, StageStorage, Stages, Tile, Walk, ops::softmax::OnlineSoftmax,
};
use cubecl::{self as cubecl, cube, prelude::*};

/// A cube's planes running the plane fragment microkernel over the cube's stages: each plane's
/// sum of its rows' outputs in fragments held across the walk, its score where the microkernel
/// keeps it, and the softmax state of its rows, kept where the score is.
#[derive(CubeType)]
pub struct PlaneFragmentAttention<EA: Float, ES: Float> {
    sum: Tile,
    score: StageScore,
    softmax: OnlineSoftmax<EA>,
    query: Tile,
    /// The element the keys and values are staged at, which the probabilities are cast to.
    #[cube(comptime)]
    staged: PhantomData<ES>,
    #[cube(comptime)]
    microkernel: PlaneFragmentAttentionMicrokernel,
    #[cube(comptime)]
    layout: CubePlaneLayout,
}

#[cube]
impl<EA: Float, E: Float, ES: Float> CubePlaneAttention<EA, E, Tile, Tile, Tile>
    for PlaneFragmentAttention<EA, ES>
{
    type Microkernel = PlaneFragmentAttentionMicrokernel;
    type Opening = StagedPlaneOperands;
    type Line = Vector<EA, Const<1>>;

    /// Its score placed by the walk, and its sum opened at zero on the cube's output.
    fn open(
        walk: &Walk,
        opening: &StagedPlaneOperands,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
        #[comptime] layout: CubePlaneLayout,
    ) -> Self {
        let score = StageScore::open::<EA, ES>(walk, &opening.query, microkernel);
        let softmax = score.softmax::<EA>(comptime!(microkernel.queries()));
        PlaneFragmentAttention::<EA, ES> {
            sum: plane_fragment_attention_sum::<EA>(
                &opening.out,
                &opening.bias,
                &opening.value,
                microkernel,
            ),
            score,
            softmax,
            query: opening.query.clone(),
            staged: PhantomData,
            microkernel,
            layout,
        }
    }

    /// The plane's part of the stage taken into its rows ([`StageScore::step`]).
    fn step(&mut self, plane: &Region, keys: &Tile, values: &Tile, visible: &Tile, scale: EA) {
        let mut sum = self.sum.at(plane);
        self.score.at(plane).step::<EA, ES>(
            plane,
            &mut sum,
            &mut self.softmax,
            &self.query.at(plane),
            &keys.at(plane),
            &values.at(plane),
            &visible.at(plane),
            scale,
            self.microkernel,
            comptime!(self.microkernel.score_levels(&[
                (G, self.layout.heads),
                (Q, self.layout.planes_along_queries),
                (S, self.layout.planes_along_keys),
            ])),
        );
    }

    /// Every plane's rows read off the walk's first step, whether or not the walk took one: what
    /// the softmax and the output read how many planes share a plane's rows from. The output sums
    /// the planes that split the same rows' keys.
    fn store_normalized(&mut self, cube: &Region, out: &Tile) {
        let mut output = out.planes_output::<EA>(cube);
        for plane in cube.planes() {
            let sum = self.sum.at(&plane);
            sum.along(DV).mul(&self.softmax.normalizer(&plane));
            output.drain(&sum, &plane);
        }
        output.write();
    }

    /// The plane's sum drained, as it stands, into its window of a scratch of the walk's step,
    /// and its rows' softmax state into two more, a cell a row, each written by a unit that holds
    /// it ([`OnlineSoftmax::store_rows`]); then each row's elements copied from there into its
    /// slot by the plane's units in turn, beside its state.
    ///
    /// The scratches hold the cube's rows once, so planes holding the same rows would share them:
    /// the cube that stores this way holds one plane along the keys.
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        walk: &Walk,
        plane: &Region,
        slot_out: &mut [Vector<EA, Const<1>>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    ) {
        let window = Tile::scratch::<EA>(walk, comptime!(vec![G, Q, DV]), StageStorage::Strided);
        let rows_window = window.at(plane);
        self.sum.at(plane).drained_into(&rows_window);
        let mut max =
            Tile::scratch::<EA>(walk, comptime!(vec![G, Q]), StageStorage::Strided).at(plane);
        let mut sum =
            Tile::scratch::<EA>(walk, comptime!(vec![G, Q]), StageStorage::Strided).at(plane);
        self.softmax
            .store_rows(&self.score.at(plane).scores(), &mut max, &mut sum);
        // The fragments' cells and the rows' state land in the windows before they are read.
        sync_plane();
        let value = comptime!(rows_window.space().extent(DV));
        let read = rows_window.flat::<EA, Const<1>>();
        let (max, sum) = (max.flat::<EA, Const<1>>(), sum.flat::<EA, Const<1>>());
        #[unroll]
        for r in 0..comptime!(self.microkernel.queries()) {
            if r < rows {
                let slot = first_slot + r * row_stride;
                let mut d = UNIT_POS_PLANE as usize;
                while d < value {
                    slot_out[slot * value + d] = read.read(r * value + d);
                    d += PLANE_DIM as usize;
                }
                if UNIT_POS_PLANE == 0 {
                    slot_max[slot] = max.read(r).extract(0usize);
                    slot_sum[slot] = sum.read(r).extract(0usize);
                }
            }
        }
    }
}

/// A stage's score as its plane keeps it between the two multiplications ([`ScoreHolder`]): the
/// plane's windows of the scores and the probabilities, or the score's accumulator fragments.
// A cube type lives while the kernel expands, a handle a variant: its size is never on a device.
#[allow(clippy::large_enum_variant)]
#[derive(CubeType, Clone)]
pub enum StageScore {
    Window(Tile, Tile),
    Fragments(Tile),
}

#[cube]
impl StageScore {
    /// The score of one stage of `walk` for every plane, where `microkernel` keeps it: windows of
    /// shared memory, the scores at `EA` and the probabilities at `E`, or fragments at `EA` opened
    /// against the staged `query`, which hold no memory.
    pub fn open<EA: Float, E: Float>(
        walk: &Walk,
        query: &Tile,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    ) -> StageScore {
        match comptime!(microkernel.score_holder()) {
            ScoreHolder::Window => StageScore::new_Window(
                Tile::scratch::<EA>(walk, comptime!(vec![G, Q, S]), StageStorage::Strided),
                Tile::scratch::<E>(walk, comptime!(vec![G, Q, S]), StageStorage::Strided),
            ),
            ScoreHolder::Fragments => StageScore::new_Fragments(Tile::stage_accumulator::<EA>(
                walk,
                comptime!(vec![G, Q, S]),
                query,
                comptime!(microkernel.score.tile_instruction()),
            )),
        }
    }

    /// The tile the scores are kept in: what the softmax state of its rows is taken in from.
    pub fn scores(&self) -> Tile {
        match self {
            StageScore::Window(scores, _) => scores.clone(),
            StageScore::Fragments(logits) => logits.clone(),
        }
    }

    /// `plane`'s part of the score.
    pub fn at(&self, plane: &Region) -> StageScore {
        match self {
            StageScore::Window(scores, probabilities) => {
                StageScore::new_Window(scores.at(plane), probabilities.at(plane))
            }
            StageScore::Fragments(logits) => StageScore::new_Fragments(logits.at(plane)),
        }
    }

    /// The softmax state of a plane's `queries` rows, kept where the score is: every row of the
    /// window by every unit, or by each unit the rows it holds fragment cells of.
    pub fn softmax<EA: Float>(&self, #[comptime] queries: usize) -> OnlineSoftmax<EA> {
        match self {
            StageScore::Window(_, _) => OnlineSoftmax::<EA>::new(queries, S),
            StageScore::Fragments(logits) => OnlineSoftmax::<EA>::along(logits, S),
        }
    }

    /// One stage of one plane's rows: the scores of its keys, taken into the rows' softmax, and
    /// the probabilities contracted with its values into `sum`, the score kept where this one
    /// holds it.
    ///
    /// `stage` is the plane's region of the stage; this score, `sum`, `query`, `key`, `value` and
    /// `bias` are windowed to the plane. `score_levels` is the score's own grid under the walker's
    /// planes ([`PlaneFragmentAttentionMicrokernel::score_levels`]), which a score read through
    /// windows is placed in.
    #[allow(clippy::too_many_arguments)]
    pub fn step<EA: Float, E: Float>(
        &self,
        stage: &Region,
        sum: &mut Tile,
        softmax: &mut OnlineSoftmax<EA>,
        query: &Tile,
        key: &Tile,
        value: &Tile,
        bias: &Tile,
        scale: EA,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
        #[comptime] score_levels: Vec<Level>,
    ) {
        match self {
            StageScore::Window(scores, probabilities) => Self::window_step::<EA>(
                stage,
                sum,
                scores,
                &mut probabilities.clone(),
                softmax,
                query,
                key,
                value,
                bias,
                scale,
                microkernel,
                score_levels,
            ),
            StageScore::Fragments(logits) => Self::fragments_step::<EA, E>(
                stage,
                sum,
                logits,
                softmax,
                query,
                key,
                value,
                bias,
                scale,
                microkernel,
            ),
        }
    }

    /// [`ScoreHolder::Window`]: the score contracted on its own grid, `score_levels`, and cast into
    /// the plane's window, the softmax read there, and the probabilities cast into their window for
    /// the value's fragments to load. The plane meets on `sync_plane` between the score landing and
    /// the softmax reading it, and between the probabilities landing and the value's fragments
    /// reading them.
    #[allow(clippy::too_many_arguments)]
    fn window_step<EA: Float>(
        stage: &Region,
        sum: &mut Tile,
        scores: &Tile,
        probabilities: &mut Tile,
        softmax: &mut OnlineSoftmax<EA>,
        query: &Tile,
        key: &Tile,
        value: &Tile,
        bias: &Tile,
        scale: EA,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
        #[comptime] score_levels: Vec<Level>,
    ) {
        let wide = ComptimeOption::<Tile>::new_None();
        // The score's accumulator, cut to its own grid below the planes' level.
        let score_window = scores.with_levels(1usize, score_levels);
        let mut logits = score_window.accumulator::<EA>(
            query,
            key,
            comptime!(microkernel.score.tile_instruction()),
            Semiring::SUM_PROD,
        );
        logits.zero();
        // Each operand's fragments, walked over its own box and paired by position: the query's
        // rows by steps of the head dim, the keys' by the same steps, the scores' rows by keys.
        let [(_, m), (_, n), (_, k)] = comptime!(microkernel.score.instruction.shape());
        let (rows, columns, depth) = comptime!((
            microkernel.score.settings.along_m.get(),
            microkernel.score.settings.along_n.get(),
            microkernel.score.settings.depth.get(),
        ));
        let query_fragments = query.over(&comptime!(Level::every(&[(Q, m), (D, k)])));
        let key_fragments = key.over(&comptime!(Level::every(&[(S, n), (D, k)])));
        let score_fragments = score_window.over(&comptime!(Level::every(&[(Q, m), (S, n)])));
        #[unroll]
        for step in 0..depth {
            #[unroll]
            for row in 0..rows {
                #[unroll]
                for column in 0..columns {
                    let mut fragment = logits.at(&score_fragments.region(row * columns + column));
                    fragment.mma(
                        &query.at(&query_fragments.region(row * depth + step)),
                        &key.at(&key_fragments.region(column * depth + step)),
                    );
                }
            }
        }
        // The accumulator's fragments and the scores' window are one grid at two depths: the
        // accumulator is placed on the score's own grid, the window sits under the plane's
        // region, and a region only windows a tile at its depth.
        let window_fragments = scores.over(&comptime!(Level::every(&[(Q, m), (S, n)])));
        #[unroll]
        for fragment in 0..comptime!(rows * columns) {
            let mut window = scores.at(&window_fragments.region(fragment));
            window.copy_cast_from(&logits.at(&score_fragments.region(fragment)));
        }
        // The score's cells were stored in the fragments' layout; the softmax reads them by rows.
        sync_plane();
        let correction = softmax.step(scores, bias, scale);
        probabilities.copy_cast_from(scores);
        // The probabilities land before the fragments load them.
        sync_plane();
        sum.along(DV).mul(&correction);
        // A tile's clone is a handle to the same cells: the shared microkernel reads the
        // probabilities and adds into the sum through it.
        plane_fragment_microkernel(
            stage,
            &sum.clone(),
            &wide,
            &probabilities.clone(),
            value,
            comptime!(microkernel.value.tile_instruction()),
            comptime!(None),
            comptime!(MatmulOperands::all(MatmulRead::AsBound)),
        );
    }

    /// [`ScoreHolder::Fragments`]: the score contracted into its fragments, the softmax read where
    /// the plane's units hold their cells, and the fragments, cast in their registers, contracted
    /// with the values. Nothing crosses the plane's units but the row reductions' shuffles, so the
    /// plane never meets.
    #[allow(clippy::too_many_arguments)]
    fn fragments_step<EA: Float, E: Float>(
        stage: &Region,
        sum: &mut Tile,
        logits: &Tile,
        softmax: &mut OnlineSoftmax<EA>,
        query: &Tile,
        key: &Tile,
        value: &Tile,
        bias: &Tile,
        scale: EA,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    ) {
        let wide = ComptimeOption::<Tile>::new_None();
        let mut logits = logits.clone();
        logits.reset();
        plane_fragment_microkernel(
            stage,
            &logits,
            &wide,
            query,
            key,
            comptime!(microkernel.score.tile_instruction()),
            comptime!(Some(DV)),
            comptime!(MatmulOperands::all(MatmulRead::AsBound)),
        );
        let correction = softmax.step(&logits, bias, scale);
        sum.along(DV).mul(&correction);
        plane_fragment_microkernel(
            stage,
            &sum.clone(),
            &wide,
            &logits.cast::<E>(),
            value,
            comptime!(microkernel.value.tile_instruction()),
            comptime!(Some(D)),
            comptime!(MatmulOperands::all(MatmulRead::AsBound)),
        );
    }
}

/// The output's accumulator every stage's product adds into, at `EA`, opened at zero on `out`
/// against the value: a cmma grid with the scratch its rows are scaled through, a manual one as it
/// is, which scales its rows in place.
#[cube]
fn plane_fragment_attention_sum<EA: Float>(
    out: &Tile,
    bias: &Tile,
    value: &Tile,
    #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
) -> Tile {
    let opened = out.accumulator::<EA>(
        bias,
        value,
        comptime!(microkernel.tile_instruction()),
        Semiring::SUM_PROD,
    );
    let mut sum = match comptime!(microkernel.score_holder()) {
        ScoreHolder::Window => opened.with_scratch(microkernel.settings.scratch),
        ScoreHolder::Fragments => opened,
    };
    sum.zero();
    sum
}

/// One plane's rows: the plane stages its query once and its keys and values a stage at a time,
/// holding as many stages as `ring` says, runs the plane fragment microkernel on every stage of its
/// first `steps` steps, and drains its rows into `out` normalized. `out`, `query`, `key`, `value`
/// and `bias` are windowed to `rows`; `planes` is how many planes along the queries share the
/// cube, which a score read through windows is placed under.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_attention_microkernel<EA: Float, ES: Float>(
    rows: &Region,
    out: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
) {
    let walk = rows.walk().range(0, steps);
    let mut sum = plane_fragment_attention_sum::<EA>(out, bias, value, microkernel);
    // The query spans no key: staged once, before the walk, as the plane's own copy.
    let mut staged_query = query.stage_for(
        &walk,
        StageStorage::Strided,
        comptime!(microkernel.decoded_stage()),
    );
    staged_query.copy_from(query);
    sync_plane();
    let score = StageScore::open::<EA, ES>(&walk, &staged_query, microkernel);
    let mut softmax = score.softmax::<EA>(comptime!(microkernel.queries()));
    let mut stages = Stages::smem(
        &walk,
        key,
        value,
        StageStorage::Strided,
        comptime!(ring.stages_in_flight.get()),
        comptime!(microkernel.decoded_stage()),
    );
    stages.walk(walk, comptime!(ring.prefetch), |slot, stage| {
        let bias_stage = bias.at(stage);
        slot.consume(|key, value| {
            score.step::<EA, ES>(
                stage,
                &mut sum,
                &mut softmax,
                &staged_query,
                key,
                value,
                &bias_stage,
                scale,
                microkernel,
                comptime!(microkernel.score_levels(&[(Q, planes)])),
            );
        });
    });
    sum.along(DV).mul(&softmax.recip_l());
    sum.drained_into(out);
}
