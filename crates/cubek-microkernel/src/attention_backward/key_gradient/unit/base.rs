use crate::attention::{D, DV, G, Q, S};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use crate::matmul::MatmulOperand;
use crate::matmul::dense::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Keys of one key head a unit owns whole: for each head of the group, a step of queries at a
/// time, their scores held keys first, the probabilities read back from each query's logsumexp,
/// the gradient against the scores, and the keys' and the values' gradients summed over every
/// step of every head, all in its registers.
///
/// Its four multiplications are the shared unit microkernel's: the score, `key · queryᵀ`, its keys
/// by the step's queries through the whole head dim; the gradient against the probabilities,
/// `value · out_gradientᵀ`, the same block through the whole value dim; the values' gradient,
/// `probabilities · out_gradient`, its keys by the whole value dim through the step's queries; and
/// the keys' gradient, `score_gradient · query`, its keys by the whole head dim through them. It
/// asks nothing of the hardware, so it is its settings and those four blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitKeyGradientMicrokernel {
    pub settings: UnitKeyGradientMicrokernelSettings,
    /// The score's block: keys by a step's queries, the whole head dim deep.
    pub score: UnitRegistersMicrokernel,
    /// The block of the gradient against the probabilities: the score's, the whole value dim deep.
    pub score_gradient: UnitRegistersMicrokernel,
    /// The values' gradient's block: keys by the whole value dim, a step's queries deep.
    pub value_gradient: UnitRegistersMicrokernel,
    /// The keys' gradient's block: keys by the whole head dim, a step's queries deep.
    pub key_gradient: UnitRegistersMicrokernel,
}

/// The keys a unit owns, and the queries one step of its walk scores them against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitKeyGradientMicrokernelSettings {
    pub seq_kv: NonZeroUsize,
    pub seq_q: NonZeroUsize,
}

impl UnitKeyGradientMicrokernelSettings {
    pub const DEFAULT: Self = Self::new(1, 8);

    /// `keys` keys a unit against `queries` queries a step.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn new(keys: usize, queries: usize) -> Self {
        Self {
            seq_kv: NonZeroUsize::new(keys).unwrap(),
            seq_q: NonZeroUsize::new(queries).unwrap(),
        }
    }
}

impl core::fmt::Display for UnitKeyGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "keys={} queries={}", self.seq_kv, self.seq_q)
    }
}

impl UnitKeyGradientMicrokernel {
    /// The levels inside one unit's keys, which a routine states the rest of the partitioning
    /// around: one step of queries of one head against the keys, the whole head and value dims,
    /// walked through all of `Q`, and that walk through every head of the group. Below a step
    /// lies a walk through `D`, one step of the whole head dim, inside which every multiplication
    /// contracts its block.
    pub fn tiling(&self) -> Levels {
        Levels::leaf(&[
            (S, self.seq_kv()),
            (G, 1),
            (Q, self.seq_q()),
            (D, self.score.settings.depth.get()),
            (DV, self.score_gradient.settings.depth.get()),
        ])
        .walk_every(&[D])
        .walk_every(&[Q])
        .walk_every(&[G])
    }

    /// Keys a unit owns.
    pub fn seq_kv(&self) -> usize {
        self.settings.seq_kv.get()
    }

    /// Queries one step scores the keys against.
    pub fn seq_q(&self) -> usize {
        self.settings.seq_q.get()
    }
}

/// Its keys and queries.
impl core::fmt::Display for UnitKeyGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl Microkernel for UnitKeyGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = UnitKeyGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type Partitioning = Partitioning;

    const DECLINE: &'static str = "the unit microkernel asks nothing of the hardware";

    /// The default, then two keys a unit against eight queries, and a key a unit against four and
    /// sixteen.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [(1, 8), (2, 8), (1, 4), (1, 16)]
            .map(|(keys, queries)| Self::Settings::new(keys, queries))
            .into()
    }

    /// The shared unit microkernel for each multiplication, at the blocks the settings and the
    /// dims the item holds size. It asks nothing of the hardware, so every device runs it; an item
    /// with no head or value dim has nothing to contract.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let settings = *settings;
        let dims = NonZeroUsize::new(item.head_dim).zip(NonZeroUsize::new(item.val_dim));
        let multiplications = dims.and_then(|(head, value)| {
            let block = |columns, depth| UnitRegistersMicrokernelSettings {
                rows: settings.seq_kv,
                columns,
                depth,
            };
            let queries = settings.seq_q;
            let score = UnitRegistersMicrokernel::best(hw, &item.score(S), &block(queries, head));
            let score_gradient =
                UnitRegistersMicrokernel::best(hw, &item.score_gradient(S), &block(queries, value));
            let value_gradient = UnitRegistersMicrokernel::best(
                hw,
                &item.value_gradient(item.sum),
                &block(value, queries),
            );
            let key_gradient = UnitRegistersMicrokernel::best(
                hw,
                &item.key_gradient(item.sum),
                &block(head, queries),
            );
            Some((
                score.ok()?,
                score_gradient.ok()?,
                value_gradient.ok()?,
                key_gradient.ok()?,
            ))
        });
        multiplications
            .map(
                |(score, score_gradient, value_gradient, key_gradient)| Self {
                    settings,
                    score,
                    score_gradient,
                    value_gradient,
                    key_gradient,
                },
            )
            .into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score reads the keys and the query
    /// along `D`, the gradient against the probabilities the values and the output's gradient
    /// along `DV`, the values' gradient the output's gradient along `DV` and writes along `DV`, and
    /// the keys' gradient the query along `D` and writes along `D`. The logsumexp and the dot
    /// products are read a value a query.
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
        for operand in [Operand::Logsumexp, Operand::Delta] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }

    /// Each multiplication's shared microkernel sets the work of its block; the unit holds the
    /// four blocks at once. Its steps' reads are its routine's to count, which places its units.
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
            (self.value_gradient, item.value_gradient(item.sum)),
            (self.key_gradient, item.key_gradient(item.sum)),
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
