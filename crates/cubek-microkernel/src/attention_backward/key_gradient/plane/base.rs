use crate::attention::{D, DV, G, Q, S};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use crate::matmul::dense::plane::{
    PlaneFragmentInstruction, PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
    PlaneFragmentTransport,
};
use crate::matmul::{MatmulDim, MatmulOperand, MatmulWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Level, Levels, Partitioning, Scratch};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A box of keys of one key head a plane owns: for each head of the group, a stage of queries at a
/// time, their scores keys first in the plane's window, the probabilities read back there from
/// each query's logsumexp and cast for the fragments to load, the gradient against them in another
/// window, cast in turn; the values' and the keys' gradients in fragments held across every stage
/// of every head.
///
/// Its four multiplications are the shared plane fragment microkernel's, on one cmma instruction:
/// the score, `key · queryᵀ`, its keys by the stage's queries walking the head dim; the gradient
/// against the probabilities, `value · out_gradientᵀ`, the same grid walking the value dim; the
/// values' gradient, `probabilities · out_gradient`, its keys by the value dim walking the stage's
/// queries; and the keys' gradient, `score_gradient · query`, its keys by the head dim walking
/// them. The keys' gradient is the walk's; the other three are contracted on their own grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneKeyGradientMicrokernel {
    pub settings: PlaneKeyGradientMicrokernelSettings,
    /// The score's fragments: its keys by a stage's queries, walking the whole head dim.
    pub score: PlaneFragmentMicrokernel,
    /// The fragments of the gradient against the probabilities: the score's grid, walking the
    /// whole value dim.
    pub score_gradient: PlaneFragmentMicrokernel,
    /// The values' gradient's fragments: its keys by the whole value dim, walking a stage's queries.
    pub value_gradient: PlaneFragmentMicrokernel,
    /// The keys' gradient's fragments: its keys by the whole head dim, walking a stage's queries.
    pub key_gradient: PlaneFragmentMicrokernel,
}

/// The instructions a plane's box holds along the keys and along a stage of queries, and how much
/// of the keys' gradient its scale bounces through shared memory at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneKeyGradientMicrokernelSettings {
    pub seq_kv: NonZeroUsize,
    pub seq_q: NonZeroUsize,
    pub scratch: Scratch,
}

impl PlaneKeyGradientMicrokernelSettings {
    pub const DEFAULT: Self = Self::fragments(2, 2);

    /// `keys` fragments of keys against `queries` fragments of queries, one fragment bounced at a
    /// time.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn fragments(keys: usize, queries: usize) -> Self {
        Self {
            seq_kv: NonZeroUsize::new(keys).unwrap(),
            seq_q: NonZeroUsize::new(queries).unwrap(),
            scratch: Scratch::OneTile,
        }
    }
}

impl core::fmt::Display for PlaneKeyGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fragments={}x{}", self.seq_kv, self.seq_q)?;
        match self.scratch {
            Scratch::WholeGrid => write!(f, " scratch=whole"),
            Scratch::OneTile | Scratch::None => Ok(()),
        }
    }
}

impl PlaneKeyGradientMicrokernel {
    /// The levels inside one plane's keys, which a routine states the rest of the partitioning
    /// around: a stage of queries of one head, one grid of fragments of both gradients, its keys
    /// by the head dim and the value dim and the steps of the stage's queries, each gradient
    /// walking its own of them and holding the other's; walked through all of `Q`, and that walk
    /// through every head of the group.
    pub fn tiling(&self) -> Levels {
        self.step_tiling().walk_every(&[Q]).walk_every(&[G])
    }

    /// The levels inside one stage of one plane's keys: one grid of fragments of both gradients,
    /// of one head's queries. A cube places its planes above them.
    pub fn step_tiling(&self) -> Levels {
        let [(_, keys), (_, columns), (_, queries)] = self.key_gradient.instruction.shape();
        let (key_gradient, value_gradient) =
            (self.key_gradient.settings, self.value_gradient.settings);
        Levels::leaf(&[(S, keys), (G, 1), (Q, queries), (D, columns), (DV, columns)]).walk(&[
            (S, key_gradient.along_m.get()),
            (Q, key_gradient.depth.get()),
            (D, key_gradient.along_n.get()),
            (DV, value_gradient.along_n.get()),
        ])
    }

    /// Keys one plane owns, of one key head.
    pub fn seq_kv(&self) -> usize {
        self.key_gradient.tile_extent(S)
    }

    /// Queries one stage holds.
    pub fn seq_q(&self) -> usize {
        self.key_gradient.tile_extent(Q)
    }

    /// The score's grid below the walker's level of planes, `planes` its counts.
    pub fn score_levels(&self, planes: &[(Axis, usize)]) -> Vec<Level> {
        Self::grid_levels(self.score, planes)
    }

    /// The grid of the gradient against the probabilities below the walker's level of planes.
    pub fn score_gradient_levels(&self, planes: &[(Axis, usize)]) -> Vec<Level> {
        Self::grid_levels(self.score_gradient, planes)
    }

    /// Bytes of shared memory one plane holds beside its stages: its staged keys and values, and
    /// what it walks with ([`walk_bytes`](Self::walk_bytes)).
    pub fn window_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let staged = self.seq_kv() * (item.head_dim + item.val_dim) * item.element.size();
        staged + self.walk_bytes(item)
    }

    /// Bytes of shared memory one plane holds while it walks, wherever its keys lie: its windows
    /// of the scores and of the gradient against the probabilities, the window either is cast into
    /// for the fragments, and the fragments the two gradients bounce through at once.
    pub fn walk_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let windows = self.seq_kv() * self.seq_q() * (2 * item.sum.size() + item.element.size());
        let bounced = self
            .key_gradient
            .scratch_bytes(&item.key_gradient(item.element), self.settings.scratch)
            + self
                .value_gradient
                .scratch_bytes(&item.value_gradient(item.element), self.settings.scratch);
        windows + bounced
    }

    /// `product`'s own levels below the walker's planes, `planes` its counts: its fragments, its
    /// rows by its columns by the steps of its contraction.
    fn grid_levels(product: PlaneFragmentMicrokernel, planes: &[(Axis, usize)]) -> Vec<Level> {
        let [(_, m), (_, n), (_, k)] = product.instruction.shape();
        let (axes, settings) = (product.axes, product.settings);
        Levels::leaf(&[(axes.m, m), (axes.n, n), (axes.k, k)])
            .walk(&[
                (axes.m, settings.along_m.get()),
                (axes.n, settings.along_n.get()),
                (axes.k, settings.depth.get()),
            ])
            .planes(planes)
            .build()
    }

    /// `product` on the score's instruction, `along_m` by `along_n` fragments, `depth` steps of its
    /// contraction.
    fn beside(
        score: PlaneFragmentMicrokernel,
        product: &MatmulWorkItem,
        (along_m, along_n, depth): (NonZeroUsize, NonZeroUsize, NonZeroUsize),
    ) -> PlaneFragmentMicrokernel {
        PlaneFragmentMicrokernel {
            instruction: score.instruction,
            orders: score.instruction.orders(product.layouts),
            settings: PlaneFragmentMicrokernelSettings {
                along_m,
                along_n,
                depth,
                ..score.settings
            },
            axes: product.axes,
            blocks: None,
        }
    }

    /// `extent` in steps of `step`, where they divide it.
    fn steps(extent: usize, step: usize) -> Option<NonZeroUsize> {
        NonZeroUsize::new(extent / step).filter(|_| extent.is_multiple_of(step))
    }
}

/// Its instruction and fragments.
impl core::fmt::Display for PlaneKeyGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.score.instruction, self.settings)
    }
}

impl Microkernel for PlaneKeyGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = PlaneKeyGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type Partitioning = Partitioning;

    const DECLINE: &'static str = "the plane backward microkernel runs on a cmma instruction of \
         the operands' element summing in f32, whose k divides the head and value dims and whose n \
         divides the head and value dims, on a device that commits to one plane width";

    /// The default, a fragment of each, more fragments along the keys or along the queries, and
    /// the whole grid of both gradients bounced at once.
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
        ]
    }

    /// The shared plane fragment microkernel's cmma candidates for the score at the settings'
    /// grid, whose instruction runs the other three beside it: its `k` divides the head and value
    /// dims, and its `n` the head and value dims. Both gradients contract the stage's queries, the
    /// score's columns, in steps of the instruction's `k`.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let settings = *settings;
        let grid = PlaneFragmentMicrokernelSettings {
            along_m: settings.seq_kv,
            along_n: settings.seq_q,
            transport: PlaneFragmentTransport::Cmma,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        };
        let (score_item, gradient_item) = (item.score(S), item.score_gradient(S));
        let value_item = item.value_gradient(item.element);
        let key_item = item.key_gradient(item.element);
        let (head, value) = (item.head_dim, item.val_dim);
        let candidates: Vec<_> = PlaneFragmentMicrokernel::candidates(hw, &score_item, &grid)
            .filter(|score| matches!(score.instruction, PlaneFragmentInstruction::Cmma { .. }))
            .filter_map(|score| {
                let [_, (_, n), (_, k)] = score.instruction.shape();
                let head_steps = Self::steps(head, k)?;
                let value_steps = Self::steps(value, k)?;
                let head_columns = Self::steps(head, n)?;
                let value_columns = Self::steps(value, n)?;
                let query_steps = Self::steps(settings.seq_q.get() * n, k)?;
                Some(Self {
                    settings,
                    score: PlaneFragmentMicrokernel {
                        settings: PlaneFragmentMicrokernelSettings {
                            depth: head_steps,
                            ..score.settings
                        },
                        ..score
                    },
                    score_gradient: Self::beside(
                        score,
                        &gradient_item,
                        (settings.seq_kv, settings.seq_q, value_steps),
                    ),
                    value_gradient: Self::beside(
                        score,
                        &value_item,
                        (settings.seq_kv, value_columns, query_steps),
                    ),
                    key_gradient: Self::beside(
                        score,
                        &key_item,
                        (settings.seq_kv, head_columns, query_steps),
                    ),
                })
            })
            .collect();
        candidates.into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score reads the keys and the query,
    /// the gradient against the probabilities the values and the output's gradient, the values'
    /// gradient the output's gradient and writes along `DV`, and the keys' gradient the query and
    /// writes along `D`. The logsumexp and the dot products are read a value a query.
    fn vectorize(
        &self,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<AttentionBackwardOperand>,
    ) {
        use AttentionBackwardOperand as Operand;
        let products = [
            (self.score, Some(Operand::Key), Some(Operand::Query), None),
            (
                self.score_gradient,
                Some(Operand::Value),
                Some(Operand::OutGradient),
                None,
            ),
            (
                self.value_gradient,
                None,
                Some(Operand::OutGradient),
                Some(Operand::ValueGradient),
            ),
            (
                self.key_gradient,
                None,
                Some(Operand::Query),
                Some(Operand::KeyGradient),
            ),
        ];
        for (microkernel, lhs, rhs, out) in products {
            let mut product = Vectorization::new();
            microkernel.vectorize(&(), &mut product);
            vectorization.restate(product, |operand| match operand {
                MatmulOperand::Lhs => lhs,
                MatmulOperand::Rhs => rhs,
                MatmulOperand::Out => out,
            });
        }
        for (operand, product) in [
            (Operand::KeyGradient, self.key_gradient),
            (Operand::ValueGradient, self.value_gradient),
        ] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: product.leaf_extent(MatmulDim::N),
            });
        }
        for operand in [Operand::Logsumexp, Operand::Delta] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }

    /// Each multiplication's shared microkernel sets the work of its fragments; the plane's units
    /// hold the four at once on the instruction they share. Its stages are its routine's to count.
    fn estimate(
        &self,
        item: &AttentionBackwardWorkItem,
        _partitioning: &Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let products = [
            (self.score, item.score(S)),
            (self.score_gradient, item.score_gradient(S)),
            (self.value_gradient, item.value_gradient(item.element)),
            (self.key_gradient, item.key_gradient(item.element)),
        ];
        let mut work = None;
        for (microkernel, product) in products {
            microkernel.estimate(&product, &(), planes, estimate);
            let block = estimate.work.take();
            work = match work {
                None => block,
                Some(held) => block.map(|block| held.beside(block)),
            };
        }
        estimate.work = work;
    }
}
