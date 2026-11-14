use crate::attention::plane::ScoreHolder;
use crate::attention::{DV, G, Q, S};
use crate::attention_backward::{
    AttentionBackwardOperand, AttentionBackwardWorkItem, BoundProduct, estimate_beside,
    reading_pick, score_pick,
};
use crate::matmul::MatmulDim;
use crate::matmul::plane::{
    PlaneFragmentDrain, PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Level, Levels, Partitioning, Scratch};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// The queries of one head a plane owns: their dot product of the output with its gradient,
/// then, a stage of keys at a time, their scores in the plane's window, the probabilities read
/// back there from each query's logsumexp, the gradient against them in another window, the
/// gradient against the scores cast for the fragments to load, and the query's gradient in
/// fragments held across the walk.
///
/// Its three multiplications are the shared plane fragment microkernel's, each on the instruction
/// the tile picks for the plane's tile of it, which it cuts into calls and walks:
/// the score, `query · keyᵀ`, its queries by the stage's keys walking the head dim; the gradient
/// against the probabilities, `out_gradient · valueᵀ`, the same grid walking the value dim; and the
/// query's gradient, `score_gradient · key`, its queries by the head dim walking the stage's keys.
/// The first two are contracted on their own grids into the windows; the third is the walk's.
/// Each of the plane's products is its tile in elements ([`PlaneUnit::Elements`]).
///
/// [`PlaneUnit::Elements`]: crate::matmul::plane::PlaneUnit::Elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneQueryGradientMicrokernel {
    pub settings: PlaneQueryGradientMicrokernelSettings,
    /// The score's fragments: its queries by a stage's keys, walking the whole head dim.
    pub score: PlaneFragmentMicrokernel,
    /// The fragments of the gradient against the probabilities: the score's grid, walking the
    /// whole value dim.
    pub score_gradient: PlaneFragmentMicrokernel,
    /// The query gradient's fragments: its queries by the whole head dim, walking a stage's keys.
    pub query_gradient: PlaneFragmentMicrokernel,
}

/// The shape of the tile a plane holds along the queries and along a stage of keys, in units of the
/// device's matrix unit for the operands ([`PlaneFragmentMicrokernel::matrix_unit`]), how much of
/// the query's gradient its scale bounces through shared memory at once, and where it keeps the
/// score ([`ScoreHolder`]): in windows of shared memory, read through cmma, or in its own manual
/// `mma` fragments. The instruction is not a setting: it follows from where the score is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneQueryGradientMicrokernelSettings {
    pub seq_q: NonZeroUsize,
    pub seq_kv: NonZeroUsize,
    pub scratch: Scratch,
    pub score: ScoreHolder,
}

impl PlaneQueryGradientMicrokernelSettings {
    pub const DEFAULT: Self = Self::fragments(2, 2);

    /// `queries` units of queries against `keys` units of keys, one fragment bounced at a time.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn fragments(queries: usize, keys: usize) -> Self {
        Self {
            seq_q: NonZeroUsize::new(queries).unwrap(),
            seq_kv: NonZeroUsize::new(keys).unwrap(),
            scratch: Scratch::OneTile,
            score: ScoreHolder::Window,
        }
    }

    /// These settings on the manual `mma`, the score kept in its fragments.
    pub const fn held(self) -> Self {
        Self {
            score: ScoreHolder::Fragments,
            ..self
        }
    }
}

impl core::fmt::Display for PlaneQueryGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fragments={}x{}", self.seq_q, self.seq_kv)?;
        if self.score == ScoreHolder::Fragments {
            write!(f, " held")?;
        }
        match self.scratch {
            Scratch::WholeGrid => write!(f, " scratch=whole"),
            Scratch::OneTile | Scratch::None => Ok(()),
        }
    }
}

impl PlaneQueryGradientMicrokernel {
    /// The levels inside one plane's queries, which a routine states the rest of the partitioning
    /// around: a stage of keys, the query gradient's tiling of its fragments of queries by the head
    /// dim and the steps of the stage's keys, each spanning the value dim whole, walked through all
    /// of `S`.
    pub fn tiling(&self) -> Levels {
        self.step_tiling().walk_every(&[S])
    }

    /// The levels inside one stage of one plane's queries: the query gradient's tiling of its
    /// fragments, one head, the value dim whole. A cube places its planes above them.
    pub fn step_tiling(&self) -> Levels {
        self.query_gradient
            .tiling_holding(&[(G, 1), (DV, self.score_gradient.tile_extent(DV))])
    }

    /// Queries one plane owns, of one head.
    pub fn seq_q(&self) -> usize {
        self.query_gradient.tile_extent(Q)
    }

    /// Keys one stage holds.
    pub fn seq_kv(&self) -> usize {
        self.query_gradient.tile_extent(S)
    }

    /// The score's grid below the walker's level of planes, `planes` its counts
    /// (`grid_levels`).
    pub fn score_levels(&self, planes: &[(Axis, usize)]) -> Vec<Level> {
        Self::grid_levels(self.score, planes)
    }

    /// The grid of the gradient against the probabilities below the walker's level of planes.
    pub fn score_gradient_levels(&self, planes: &[(Axis, usize)]) -> Vec<Level> {
        Self::grid_levels(self.score_gradient, planes)
    }

    /// Where a plane keeps a step's score and the gradient against its probabilities: its
    /// settings', which its instruction was picked for.
    pub fn holder(&self) -> ScoreHolder {
        self.settings.score
    }

    /// The scratch the query gradient's sum bounces through as its scale is taken: the
    /// settings', for a sum that drains by bouncing through shared memory, a cmma one; none for
    /// one whose units hold their own cells and scale them in place.
    pub fn sum_scratch(&self) -> Scratch {
        match self.query_gradient.instruction.drain() {
            PlaneFragmentDrain::Bounce => self.settings.scratch,
            PlaneFragmentDrain::CellByCell => Scratch::None,
        }
    }

    /// Bytes of shared memory one plane holds beside its stages: its staged query and output
    /// gradient, and what it walks with ([`walk_bytes`](Self::walk_bytes)).
    pub fn window_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let staged = self.seq_q() * (item.head_dim + item.val_dim) * item.element.size();
        staged + self.walk_bytes(item)
    }

    /// Bytes of shared memory one plane holds while it walks, wherever its query lies: its windows
    /// of the scores, of the gradient against the probabilities and of that gradient cast for the
    /// fragments, and the fragments the query gradient's scale bounces through at once.
    pub fn walk_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let windows = match self.holder() {
            ScoreHolder::Window => {
                self.seq_q() * self.seq_kv() * (2 * item.sum.size() + item.element.size())
            }
            ScoreHolder::Fragments => 0,
        };
        let bounced = self
            .query_gradient
            .scratch_bytes(&item.query_gradient(item.element), self.sum_scratch());
        windows + bounced
    }

    /// `product`'s own levels below the walker's planes, `planes` its counts: the plane's tile of
    /// it, its rows by its columns by its whole contraction, which the tile cuts and walks, so the
    /// plane's tile spans the contraction whole, as the operands' windows do.
    pub(crate) fn grid_levels(
        product: PlaneFragmentMicrokernel,
        planes: &[(Axis, usize)],
    ) -> Vec<Level> {
        let axes = product.axes;
        Levels::leaf(&[
            (axes.m, product.tile_extent(axes.m)),
            (axes.n, product.tile_extent(axes.n)),
            (axes.k, product.tile_extent(axes.k)),
        ])
        .planes(planes)
        .build()
    }
}

/// Its instruction and fragments.
impl core::fmt::Display for PlaneQueryGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.score.instruction, self.settings)
    }
}

impl Microkernel for PlaneQueryGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = PlaneQueryGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type CubeSpace = Partitioning;

    const DECLINE: &'static str = "the plane backward microkernel runs on a cmma instruction, or \
         a manual mma of 16-bit operands whose accumulator the device reads as its A, of the \
         operands' element summing in f32, whose k divides the head and value dims and a stage's \
         keys and whose n divides the head dim, on a device that commits to one plane width";

    /// The default, a fragment of each, more fragments along the queries or along the keys, and
    /// the whole grid of the query gradient bounced at once; then the default, the wider grids and
    /// a fragment of queries against four of keys, the score held in its fragments.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::fragments(1, 1),
            Self::Settings::fragments(4, 2),
            Self::Settings::fragments(2, 4),
            Self::Settings {
                scratch: Scratch::WholeGrid,
                ..Self::Settings::DEFAULT
            },
            Self::Settings::DEFAULT.held(),
            Self::Settings::fragments(4, 2).held(),
            Self::Settings::fragments(2, 4).held(),
            Self::Settings::fragments(1, 4).held(),
            Self::Settings::fragments(1, 2).held(),
        ]
    }

    /// The tile's picks for the plane's three products, held where the settings say: the score,
    /// its queries by a stage's keys by the whole head dim; the gradient against the
    /// probabilities, the same queries and keys by the whole value dim; the query's gradient, its
    /// queries by the whole head dim, contracting the stage's keys. The queries and keys are the
    /// settings' counts of the device's matrix unit. Each is asked of the tile with the sources its
    /// kernel opens the sum on: in windows, two drained sums and a third of the gradient's cast
    /// window; held, two sums contracted again, the query's gradient reading runs of the score
    /// gradient's cells. None where the tile picks no instruction for one of them.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let settings = *settings;
        let grid = PlaneFragmentMicrokernelSettings {
            along_m: settings.seq_q,
            along_n: settings.seq_kv,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        };
        let (score_item, gradient_item) = (item.score(Q), item.score_gradient(Q));
        let query_item = item.query_gradient(item.element);
        let (head, value) = (item.head_dim, item.val_dim);
        let picked = || {
            hw.plane_dim().ok()?;
            let [queries, keys] = PlaneFragmentMicrokernel::unit_tile(hw, &score_item, &grid)?;
            let (score_tile, gradient_tile) = ([queries, keys, head], [queries, keys, value]);
            let query_tile = [queries, head, keys];
            let holder = settings.score;
            let score = score_pick(hw, holder, &score_item, score_tile)?;
            let gradient = score_pick(hw, holder, &gradient_item, gradient_tile)?;
            let query = reading_pick(hw, holder, gradient, &query_item, query_tile)?;
            Some(Self {
                settings,
                score: PlaneFragmentMicrokernel::holding(&score_item, grid, score_tile, score)?,
                score_gradient: PlaneFragmentMicrokernel::holding(
                    &gradient_item,
                    grid,
                    gradient_tile,
                    gradient,
                )?,
                query_gradient: PlaneFragmentMicrokernel::holding(
                    &query_item,
                    grid,
                    query_tile,
                    query,
                )?,
            })
        };
        picked().into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score reads the query and the keys,
    /// the gradient against the probabilities the output's gradient and the values, and the query's
    /// gradient the keys and writes along `D`. The dot product reads the output and its gradient a
    /// line of each at a time; the logsumexp and the dot products are read a value a query.
    fn vectorize(
        &self,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<AttentionBackwardOperand>,
    ) {
        use AttentionBackwardOperand as Operand;
        for product in [
            BoundProduct {
                microkernel: self.score,
                lhs: Some(Operand::Query),
                rhs: Some(Operand::Key),
                out: None,
            },
            BoundProduct {
                microkernel: self.score_gradient,
                lhs: Some(Operand::OutGradient),
                rhs: Some(Operand::Value),
                out: None,
            },
            BoundProduct {
                microkernel: self.query_gradient,
                lhs: None,
                rhs: Some(Operand::Key),
                out: Some(Operand::QueryGradient),
            },
        ] {
            product.restate(vectorization);
        }
        vectorization.require(VectorRequirement::Access {
            operand: Operand::QueryGradient,
            elements: self.query_gradient.leaf_extent(MatmulDim::N),
        });
        // A manual fragment drains one cell at a time, wherever its unit holds it.
        if self.query_gradient.instruction.drain() == PlaneFragmentDrain::CellByCell {
            vectorization.require(VectorRequirement::Elementwise {
                operand: Operand::QueryGradient,
            });
        }
        vectorization.require(VectorRequirement::Together {
            operand: Operand::Out,
            with: Operand::OutGradient,
        });
        for operand in [Operand::Logsumexp, Operand::Delta] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }

    /// Each multiplication's shared microkernel sets the work of its fragments; the plane's units
    /// hold the three at once. Its stages are its routine's to count.
    fn estimate(
        &self,
        item: &AttentionBackwardWorkItem,
        _partitioning: &Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate_beside(
            [
                (self.score, item.score(Q)),
                (self.score_gradient, item.score_gradient(Q)),
                (self.query_gradient, item.query_gradient(item.element)),
            ],
            planes,
            estimate,
        );
    }
}
