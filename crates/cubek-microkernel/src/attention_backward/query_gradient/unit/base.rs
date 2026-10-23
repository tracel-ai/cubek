use crate::attention::{D, DV, G, Q, S};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use crate::matmul::MatmulOperand;
use crate::matmul::dense::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Queries of one head a unit owns whole: their dot product of the output with its gradient, then,
/// a step of keys at a time, their scores against the step's keys, the probabilities read back
/// from each query's logsumexp, the gradient against the scores, and the query's gradient summed
/// over the walk, all in its registers.
///
/// Its three multiplications are the shared unit microkernel's: the score, `query · keyᵀ`, its
/// queries by the step's keys through the whole head dim; the gradient against the probabilities,
/// `out_gradient · valueᵀ`, the same block through the whole value dim; and the query's gradient,
/// `score_gradient · key`, its queries by the whole head dim through the step's keys. It asks
/// nothing of the hardware, so it is its settings and those three blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitQueryGradientMicrokernel {
    pub settings: UnitQueryGradientMicrokernelSettings,
    /// The score's block: queries by a step's keys, the whole head dim deep.
    pub score: UnitRegistersMicrokernel,
    /// The block of the gradient against the probabilities: the score's, the whole value dim deep.
    pub score_gradient: UnitRegistersMicrokernel,
    /// The query gradient's block: queries by the whole head dim, a step's keys deep.
    pub query_gradient: UnitRegistersMicrokernel,
}

/// The queries a unit owns, and the keys one step of its walk scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitQueryGradientMicrokernelSettings {
    pub seq_q: NonZeroUsize,
    pub seq_kv: NonZeroUsize,
}

impl UnitQueryGradientMicrokernelSettings {
    pub const DEFAULT: Self = Self::new(1, 8);

    /// `queries` queries a unit against `keys` keys a step.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn new(queries: usize, keys: usize) -> Self {
        Self {
            seq_q: NonZeroUsize::new(queries).unwrap(),
            seq_kv: NonZeroUsize::new(keys).unwrap(),
        }
    }
}

impl core::fmt::Display for UnitQueryGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "queries={} keys={}", self.seq_q, self.seq_kv)
    }
}

impl UnitQueryGradientMicrokernel {
    /// The levels inside one unit's queries, which a routine states the rest of the partitioning
    /// around: one step of keys against the queries of one head, the whole head and value dims,
    /// walked through all of `S`. Below a step lies a walk through `D`, one step of the whole
    /// head dim, inside which every multiplication contracts its block.
    pub fn tiling(&self) -> Levels {
        Levels::leaf(&[
            (Q, self.seq_q()),
            (G, 1),
            (S, self.seq_kv()),
            (D, self.score.settings.depth.get()),
            (DV, self.score_gradient.settings.depth.get()),
        ])
        .walk_every(&[D])
        .walk_every(&[S])
    }

    /// Queries a unit owns.
    pub fn seq_q(&self) -> usize {
        self.settings.seq_q.get()
    }

    /// Keys one step scores.
    pub fn seq_kv(&self) -> usize {
        self.settings.seq_kv.get()
    }
}

/// Its queries and keys.
impl core::fmt::Display for UnitQueryGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl Microkernel for UnitQueryGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = UnitQueryGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type Partitioning = Partitioning;

    const DECLINE: &'static str = "the unit microkernel asks nothing of the hardware";

    /// The default, then two queries a unit against eight keys, and a query a unit against four
    /// and sixteen.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [(1, 8), (2, 8), (1, 4), (1, 16)]
            .map(|(queries, keys)| Self::Settings::new(queries, keys))
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
                rows: settings.seq_q,
                columns,
                depth,
            };
            let score =
                UnitRegistersMicrokernel::best(hw, &item.score(Q), &block(settings.seq_kv, head));
            let score_gradient = UnitRegistersMicrokernel::best(
                hw,
                &item.score_gradient(Q),
                &block(settings.seq_kv, value),
            );
            let query_gradient = UnitRegistersMicrokernel::best(
                hw,
                &item.query_gradient(item.sum),
                &block(head, settings.seq_kv),
            );
            Some((score.ok()?, score_gradient.ok()?, query_gradient.ok()?))
        });
        multiplications
            .map(|(score, score_gradient, query_gradient)| Self {
                settings,
                score,
                score_gradient,
                query_gradient,
            })
            .into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score reads the query and the keys
    /// along `D`, the gradient against the probabilities the output's gradient and the values
    /// along `DV`, and the query's gradient the keys along `D` and writes along `D`. The dot
    /// product reads the output and its gradient a line of each at a time, so they share a width;
    /// the logsumexp and the dot products are read and written a value a query.
    fn vectorize(
        &self,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<AttentionBackwardOperand>,
    ) {
        use AttentionBackwardOperand as Operand;
        let products = [
            (self.score, Some(Operand::Query), Some(Operand::Key), None),
            (
                self.score_gradient,
                Some(Operand::OutGradient),
                Some(Operand::Value),
                None,
            ),
            (
                self.query_gradient,
                None,
                Some(Operand::Key),
                Some(Operand::QueryGradient),
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
        vectorization.require(VectorRequirement::Together {
            operand: Operand::Out,
            with: Operand::OutGradient,
        });
        for operand in [Operand::Logsumexp, Operand::Delta] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }

    /// Each multiplication's shared microkernel sets the work of its block; the unit holds the
    /// three blocks at once. Its steps' reads are its routine's to count, which places its units.
    fn estimate(
        &self,
        item: &AttentionBackwardWorkItem,
        _partitioning: &Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let products = [
            (self.score, item.score(Q)),
            (self.score_gradient, item.score_gradient(Q)),
            (self.query_gradient, item.query_gradient(item.sum)),
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
