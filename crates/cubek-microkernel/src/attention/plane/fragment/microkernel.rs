use super::{PlaneFragmentAttentionMicrokernel, ScoreHolder};
use crate::attention::{DV, G, Q, S};
use crate::matmul::MatmulOperands;
use crate::matmul::plane::plane_fragment_microkernel;
use core::marker::PhantomData;
use cubecl::tile::ops::softmax::OnlineSoftmax;
use cubecl::tile::{Axis, Factor, Level, Region, Semiring, StageStorage, Stages, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

/// The keys and values a plane fragment attention stages, without the scales the softmax and the
/// output's sum take out of them, and those scales: the stages hold what is taken out as stored,
/// filled by plain copies, and each scale is applied once instead.
///
/// The softmax reads every score under one factor ([`OnlineSoftmax::step`]), wherever the score
/// is held, so the keys give up the scales that vary along no axis ([`Tile::factor_whole`]). The
/// values give up what the output's sum takes out ([`Tile::factor`]), the instruction it contracts
/// through deciding; what a side keeps, its stage's fill decodes.
#[derive(CubeType, Clone)]
pub struct StagedCache {
    pub key: Tile,
    pub value: Tile,
    /// The keys' scale, which the softmax reads the score under.
    pub key_factor: Factor,
    /// The values' scale the output's sum took out, which the sum drains under.
    pub value_factor: Factor,
}

#[cube]
impl StagedCache {
    /// `key` and `value` without the scales the softmax and `sum`, the output's, take out of them.
    pub fn new(key: &Tile, value: &Tile, sum: &Tile) -> StagedCache {
        let (key, key_factor) = key.factor_whole();
        let (value, value_factor) = value.factor(sum);
        StagedCache {
            key,
            value,
            key_factor,
            value_factor,
        }
    }
}

/// Planes running the plane fragment microkernel over staged keys and values: each plane's sum of
/// its rows' outputs in fragments held across the walk, its score where the microkernel keeps it,
/// and the softmax state of its rows, kept where the score is.
#[derive(CubeType)]
pub struct PlaneFragmentAttention<EA: Float, EV: Float> {
    sum: Tile,
    score: StageScore,
    /// The keys and values its stages are filled from, and the scales its sums take on.
    pub(crate) cache: StagedCache,
    /// The softmax state of every plane's rows: what the cube merges where its planes split the
    /// same rows' keys ([`OnlineSoftmax::normalizer`]).
    pub(crate) softmax: OnlineSoftmax<EA>,
    query: Tile,
    /// The element the values are read in, which the probabilities are cast to.
    #[cube(comptime)]
    probabilities: PhantomData<EV>,
    #[cube(comptime)]
    microkernel: PlaneFragmentAttentionMicrokernel,
    /// The planes its walker lays it over, along each axis: what the score's own grid is placed
    /// under ([`PlaneFragmentAttentionMicrokernel::score_levels`]).
    #[cube(comptime)]
    planes: Vec<(Axis, usize)>,
}

#[cube]
impl<EA: Float, EV: Float> PlaneFragmentAttention<EA, EV> {
    /// Every plane's rows over `walk` before any key: its score placed by the walk, and its sum
    /// opened at zero on `out`, the output it is shaped by, against `value` and the score's
    /// probabilities, and `key` and `value` without the scales the softmax and the sum take out of
    /// them, which its stages are filled from ([`StagedCache`]). `query` is staged, and `planes` the planes
    /// its walker lays it over along each axis.
    pub fn open(
        walk: &Walk,
        query: &Tile,
        out: &Tile,
        key: &Tile,
        value: &Tile,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
        #[comptime] planes: Vec<(Axis, usize)>,
    ) -> Self {
        let score = StageScore::open::<EA, EV>(walk, query, microkernel);
        let softmax = score.softmax::<EA>(microkernel.seq_q());
        let sum = Self::open_sum(out, &score, value, microkernel);
        let cache = StagedCache::new(key, value, &sum);
        PlaneFragmentAttention::<EA, EV> {
            sum,
            score,
            cache,
            softmax,
            query: query.clone(),
            probabilities: PhantomData,
            microkernel,
            planes,
        }
    }

    /// `plane`, its region of one step of the walk, taken into its rows ([`StageScore::step`]):
    /// its keys and values read from `keys` and `values`, each row seeing those `visible` admits,
    /// scaled by `scale`.
    pub fn step(&mut self, plane: &Region, keys: &Tile, values: &Tile, visible: &Tile, scale: EA) {
        let mut sum = self.sum.at(plane);
        self.score.at(plane).step::<EA, EV>(
            plane,
            &mut sum,
            &mut self.softmax,
            &self.query.at(plane),
            &keys.at(plane),
            &self.cache.key_factor,
            &values.at(plane),
            &visible.at(plane),
            scale,
            comptime!(self.microkernel.score_levels(&self.planes)),
        );
    }

    /// The output's accumulator every stage's product adds into, at `EA`, opened at zero on `out`
    /// against what each stage contracts with the value, `score`'s probabilities: a window's sum
    /// through the fragment API with the scratch its rows are scaled through, the fragments' sum
    /// in their own instruction, which scales its rows in place.
    fn open_sum(
        out: &Tile,
        score: &StageScore,
        value: &Tile,
        #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    ) -> Tile {
        let opened = out.accumulator::<EA>(&score.probabilities(), value, Semiring::SUM_PROD);
        let mut sum = match microkernel.score_holder() {
            ScoreHolder::Window => opened.with_scratch(microkernel.settings.scratch),
            ScoreHolder::Fragments => opened,
        };

        sum.zero();
        sum
    }

    /// `plane`'s rows of the sum its stages' products add into, undivided, under the values'
    /// scale, which it is multiplied by as it drains.
    pub fn sum_at(&self, plane: &Region) -> Tile {
        self.sum.at(plane).scaled(&self.cache.value_factor)
    }

    /// Each plane's rows' logsumexp, `max + ln(sum)` of their scaled scores, written into
    /// `logsumexp`, `region`'s window of it, from their state kept where the score is
    /// ([`OnlineSoftmax::store_logsumexp`]).
    pub fn store_logsumexp(&self, region: &Region, logsumexp: &Tile) {
        for plane in region.planes() {
            self.softmax
                .store_logsumexp(&self.score.at(&plane).scores(), &mut logsumexp.at(&plane));
        }
    }

    /// `plane`'s own rows of `walk`'s first step left undivided in slots, the first `rows` of
    /// them: row `r`'s output, running max and running sum at slot `first_slot + r · row_stride`
    /// of `slot_out`, `slot_max` and `slot_sum`.
    ///
    /// The plane's sum is drained, as it stands, into its window of a scratch of the walk's step,
    /// multiplied by the values' scale, and its rows' softmax state, which no scale of the values
    /// enters, into two more, a cell a row, each written by a unit that holds it
    /// ([`OnlineSoftmax::store_rows`]); then each row's elements are copied from there into its
    /// slot by the plane's units in turn, beside its state. The scratches hold the walk's rows
    /// once, so planes holding the same rows would share them: its walker holds one plane along
    /// the keys where it stores this way.
    #[allow(clippy::too_many_arguments)]
    pub fn store_partial(
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
        self.sum_at(plane).drained_into(&rows_window);

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
        for r in 0..comptime!(self.microkernel.seq_q()) {
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
        match microkernel.score_holder() {
            ScoreHolder::Window => StageScore::new_Window(
                Tile::scratch::<EA>(walk, comptime!(vec![G, Q, S]), StageStorage::Strided),
                Tile::scratch::<E>(walk, comptime!(vec![G, Q, S]), StageStorage::Strided),
            ),
            ScoreHolder::Fragments => StageScore::new_Fragments(Tile::stage_accumulator::<EA>(
                walk,
                comptime!(vec![G, Q, S]),
                query,
            )),
        }
    }

    /// What each stage contracts with the value: the probabilities' window, or the score's own
    /// fragments, cast in their registers.
    pub fn probabilities(&self) -> Tile {
        match self {
            StageScore::Window(_, probabilities) => probabilities.clone(),
            StageScore::Fragments(logits) => logits.clone(),
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
    /// `bias` are windowed to the plane, and the softmax reads the score under `key_factor`, the
    /// keys' scale ([`StagedCache::key_factor`]). `score_levels` is the score's own grid under the
    /// walker's planes ([`PlaneFragmentAttentionMicrokernel::score_levels`]), which a score read
    /// through windows is placed in.
    #[allow(clippy::too_many_arguments)]
    pub fn step<EA: Float, E: Float>(
        &self,
        stage: &Region,
        sum: &mut Tile,
        softmax: &mut OnlineSoftmax<EA>,
        query: &Tile,
        key: &Tile,
        key_factor: &Factor,
        value: &Tile,
        bias: &Tile,
        scale: EA,
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
                key_factor,
                value,
                bias,
                scale,
                score_levels,
            ),
            StageScore::Fragments(logits) => Self::fragments_step::<EA, E>(
                stage, sum, logits, softmax, query, key, key_factor, value, bias, scale,
            ),
        }
    }

    /// [`ScoreHolder::Window`]: the score contracted on its own grid, `score_levels`, and drained
    /// into the plane's window, the softmax read there, and the probabilities cast into their
    /// window for the value's fragments to load. The plane meets on `sync_plane` between the score
    /// landing and the softmax reading it, and between the probabilities landing and the value's
    /// fragments reading them.
    #[allow(clippy::too_many_arguments)]
    fn window_step<EA: Float>(
        stage: &Region,
        sum: &mut Tile,
        scores: &Tile,
        probabilities: &mut Tile,
        softmax: &mut OnlineSoftmax<EA>,
        query: &Tile,
        key: &Tile,
        key_factor: &Factor,
        value: &Tile,
        bias: &Tile,
        scale: EA,
        #[comptime] score_levels: Vec<Level>,
    ) {
        // The score's accumulator, placed on its own grid below the planes' level: the tile cuts
        // the plane's score into its calls and walks the head dim.
        // The query and the keys are windowed to the plane as the scores are: placed on the same
        // grid, their steps along the head dim are the tile's walk of the score's.
        let score_window = scores.with_levels(1usize, score_levels.clone());
        let query = query.with_levels(1usize, score_levels.clone());
        let key = key.with_levels(1usize, score_levels);
        let mut logits = score_window.accumulator::<EA>(&query, &key, Semiring::SUM_PROD);
        logits.zero();
        logits.mma(&query, &key);
        // The window on the score's grid is the plane's window of scores: drained at the depth
        // the accumulator sits.
        logits.drained_into(&score_window);

        // The score's cells were stored in the fragments' layout; the softmax reads them by rows.
        sync_plane();

        // The keys' scale multiplies the softmax's; the window holds the probabilities after it.
        let correction = softmax.step(&scores.scaled(key_factor), bias, scale);
        probabilities.copy_cast_from(scores);

        // The probabilities land before the fragments load them.
        sync_plane();

        sum.along(DV).mul(&correction);

        // A tile's clone is a handle to the same cells: the shared microkernel reads the
        // probabilities and adds into the sum through it.
        plane_fragment_microkernel(
            stage,
            &sum.clone(),
            &probabilities.clone(),
            value,
            None,
            MatmulOperands::AS_BOUND,
        );
    }

    /// [`ScoreHolder::Fragments`]: the score contracted into its fragments, the softmax read where
    /// the plane's units hold their cells, and the fragments, cast in their registers, contracted
    /// with the values, which read runs of them as each call's `A`. Nothing crosses the plane's
    /// units but the row reductions' shuffles, so the plane never meets.
    #[allow(clippy::too_many_arguments)]
    fn fragments_step<EA: Float, E: Float>(
        stage: &Region,
        sum: &mut Tile,
        logits: &Tile,
        softmax: &mut OnlineSoftmax<EA>,
        query: &Tile,
        key: &Tile,
        key_factor: &Factor,
        value: &Tile,
        bias: &Tile,
        scale: EA,
    ) {
        let mut logits = logits.clone();
        logits.reset();

        plane_fragment_microkernel(
            stage,
            &logits,
            query,
            key,
            comptime!(Some(DV)),
            MatmulOperands::AS_BOUND,
        );

        // The keys' scale multiplies the softmax's; the fragments hold the probabilities after it.
        let correction = softmax.step(&logits.scaled(key_factor), bias, scale);
        sum.along(DV).mul(&correction);

        let probabilities = logits.cast::<E>();
        plane_fragment_microkernel(
            stage,
            &sum.clone(),
            &probabilities,
            value,
            comptime!(None),
            MatmulOperands::AS_BOUND,
        );
    }
}

/// One plane's rows: the plane stages its query once and its keys and values a stage at a time,
/// without the scales the softmax and the sum take out of them ([`StagedCache`]), holding as many
/// stages as `ring` says, runs the plane fragment microkernel on every stage of its first `steps`
/// steps, drains its rows into `out` normalized under the values' scale, and writes each row's
/// logsumexp where the caller asks for it. `out`, `logsumexp`, `query`, `key`, `value` and `bias`
/// are windowed to `rows`; `planes` is how many planes along the queries share the cube, which a
/// score read through windows is placed under.
#[cube]
pub fn plane_fragment_attention_microkernel<EA: Float, EV: Float>(
    rows: &Region,
    out: &Tile,
    logsumexp: &ComptimeOption<Tile>,
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
    let decoded = microkernel.decoded_stage();
    // The query spans no key: staged once, before the walk, as the plane's own copy.
    let mut staged_query =
        query.stage_for(&walk, StageStorage::Strided, microkernel.decoded_query());
    staged_query.copy_from(query);

    sync_plane();

    let score = StageScore::open::<EA, EV>(&walk, &staged_query, microkernel);
    let mut softmax = score.softmax::<EA>(microkernel.seq_q());
    let mut sum = PlaneFragmentAttention::<EA, EV>::open_sum(out, &score, value, microkernel);
    let cache = StagedCache::new(key, value, &sum);

    let in_flight = ring.stages_in_flight.get();
    let mut stages = Stages::smem(
        &walk,
        &cache.key,
        &cache.value,
        StageStorage::Strided,
        in_flight,
        decoded,
    );

    stages.walk(walk, ring.prefetch, |slot, stage| {
        let bias_stage = bias.at(stage);
        slot.consume(|key, value| {
            score.step::<EA, EV>(
                stage,
                &mut sum,
                &mut softmax,
                &staged_query,
                key,
                &cache.key_factor,
                value,
                &bias_stage,
                scale,
                microkernel.score_levels(&[(Q, planes)]),
            );
        });
    });

    sum.along(DV).mul(&softmax.recip_l());
    sum.scaled(&cache.value_factor).drained_into(out);

    #[comptime]
    match logsumexp {
        ComptimeOption::Some(logsumexp) => {
            softmax.store_logsumexp(&score.scores(), &mut logsumexp.clone())
        }
        ComptimeOption::None => {}
    }
}
