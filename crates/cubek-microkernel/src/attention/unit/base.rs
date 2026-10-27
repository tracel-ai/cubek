use crate::attention::AttentionMicrokernel;
use crate::attention::{
    AttentionOperand, AttentionPartitioning, AttentionWorkItem, D, DV, G, Q, S,
};
use crate::matmul::dense::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use crate::matmul::{MatmulMicrokernel, MatmulOperand, MatmulOrder, MatmulWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::Levels;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// Rows of one head a unit owns whole: their scores against a step of keys, the softmax of every
/// row, and their outputs over the whole value dim, all in its registers while it walks the keys.
///
/// Its two multiplications are the shared unit microkernel's: the score, `query · keyᵀ`, a block
/// of its rows by the step's keys contracting the whole head dim, and the value,
/// `probabilities · value`, a block of its rows by the whole value dim contracting the step's
/// keys. It asks nothing of the hardware, so it is its settings and those two blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitAttentionMicrokernel {
    pub settings: UnitAttentionMicrokernelSettings,
    /// The score's block: rows by a step's keys, the whole head dim deep.
    pub score: UnitRegistersMicrokernel,
    /// The value's block: rows by the whole value dim, a step's keys deep.
    pub value: UnitRegistersMicrokernel,
}

/// The query rows a unit owns, and the keys one step of its walk scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitAttentionMicrokernelSettings {
    pub seq_q: NonZeroUsize,
    pub seq_kv: NonZeroUsize,
}

impl UnitAttentionMicrokernelSettings {
    pub const DEFAULT: Self = Self::new(1, 8);

    /// `rows` rows a unit against `keys` keys a step.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn new(rows: usize, keys: usize) -> Self {
        Self {
            seq_q: NonZeroUsize::new(rows).unwrap(),
            seq_kv: NonZeroUsize::new(keys).unwrap(),
        }
    }
}

impl core::fmt::Display for UnitAttentionMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "rows={} keys={}", self.seq_q, self.seq_kv)
    }
}

impl UnitAttentionMicrokernel {
    /// The levels inside one unit's rows, which a routine states the rest of the partitioning
    /// around: its [`step_tiling`](Self::step_tiling), walked through all of `S`.
    pub fn tiling(&self) -> Levels {
        self.step_tiling().walk_every(&[S])
    }

    /// The levels inside one unit's rows of one step: a step of keys against the rows of one head,
    /// the whole head and value dims. Below it lies the score's walk through `D`, one step of the
    /// whole head dim, inside which both multiplications contract their blocks.
    pub fn step_tiling(&self) -> Levels {
        Levels::leaf(&[
            (Q, self.settings.seq_q.get()),
            (G, 1),
            (S, self.seq_kv()),
            (D, self.head_dim()),
            (DV, self.val_dim()),
        ])
        .walk_every(&[D])
    }

    /// 32-bit registers one unit keeps the softmax's running max and sum of every row in, beside
    /// both multiplications' blocks.
    fn softmax_registers(&self, item: &AttentionWorkItem) -> u32 {
        (2 * self.settings.seq_q.get() * item.sum.size()).div_ceil(4) as u32
    }

    /// Keys one step scores.
    pub fn seq_kv(&self) -> usize {
        self.settings.seq_kv.get()
    }

    /// The head dim a row of the query spans: the score's depth.
    fn head_dim(&self) -> usize {
        self.score.settings.depth.get()
    }

    /// The value dim a row of the output spans: the value's columns.
    fn val_dim(&self) -> usize {
        self.value.settings.columns.get()
    }

    /// The score as the shared microkernel is handed it: summed into a register block.
    fn score_item(item: &AttentionWorkItem) -> MatmulWorkItem {
        item.score(item.sum)
    }

    /// The value as the shared microkernel is handed it: out of the score's register block.
    fn value_item(item: &AttentionWorkItem) -> MatmulWorkItem {
        item.value(item.sum)
    }
}

/// Its rows and keys.
impl core::fmt::Display for UnitAttentionMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl Microkernel for UnitAttentionMicrokernel {
    type WorkItem = AttentionWorkItem;
    type Settings = UnitAttentionMicrokernelSettings;
    type Operand = AttentionOperand;
    type Partitioning = AttentionPartitioning;

    const DECLINE: &'static str = "the unit microkernel asks nothing of the hardware";

    /// The default, then two rows a unit against eight keys, and a row a unit against four and
    /// sixteen.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [(1, 8), (2, 8), (1, 4), (1, 16)]
            .map(|(rows, keys)| Self::Settings::new(rows, keys))
            .into()
    }

    /// The shared unit microkernel for each multiplication, at the blocks the settings and the
    /// dims the item holds size. It asks nothing of the hardware, so every device runs it; an item
    /// with no head or value dim has nothing to contract.
    fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let settings = *settings;
        let dims = NonZeroUsize::new(item.head_dim).zip(NonZeroUsize::new(item.val_dim));
        let multiplications = dims.and_then(|(head, value)| {
            let score = UnitRegistersMicrokernelSettings {
                rows: settings.seq_q,
                columns: settings.seq_kv,
                depth: head,
            };
            let value = UnitRegistersMicrokernelSettings {
                rows: settings.seq_q,
                columns: value,
                depth: settings.seq_kv,
            };
            let score = UnitRegistersMicrokernel::best(hw, &Self::score_item(item), &score);
            let value = UnitRegistersMicrokernel::best(hw, &Self::value_item(item), &value);
            score.ok().zip(value.ok())
        });
        multiplications
            .map(|(score, value)| Self {
                settings,
                score,
                value,
            })
            .into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score reads the query and the keys
    /// along `D`, where the keys lie contiguous, summing a line of each into every score; the
    /// value reads the values along `DV` and writes the output along `DV`, lining its block with
    /// both. The score's block, a value a cell, is the value's lhs, read in registers.
    fn vectorize(
        &self,
        _partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        let mut score = Vectorization::new();
        self.score.vectorize(&(), &mut score);
        vectorization.restate(score, |operand| match operand {
            MatmulOperand::Lhs => Some(AttentionOperand::Query),
            MatmulOperand::Rhs => Some(AttentionOperand::Key),
            MatmulOperand::Out => None,
        });
        let mut value = Vectorization::new();
        self.value.vectorize(&(), &mut value);
        vectorization.restate(value, |operand| match operand {
            MatmulOperand::Lhs => None,
            MatmulOperand::Rhs => Some(AttentionOperand::Value),
            MatmulOperand::Out => Some(AttentionOperand::Out),
        });
    }

    /// Each multiplication's shared microkernel sets the work of its block; the unit holds both
    /// blocks at once, beside the softmax's state of every row. Its steps' reads are its routine's
    /// to count, which places its units.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        _partitioning: &AttentionPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.score
            .estimate(&Self::score_item(item), &(), planes, estimate);
        let score = estimate.work.take();
        self.value
            .estimate(&Self::value_item(item), &(), planes, estimate);
        let softmax = UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: self.softmax_registers(item),
        };
        estimate.work = score
            .zip(estimate.work)
            .map(|(score, value)| score.beside(value).beside(softmax));
    }
}

impl AttentionMicrokernel for UnitAttentionMicrokernel {
    /// The score's: a register block reads the keys in their buffer's order, along `D`.
    fn key_order(&self) -> MatmulOrder {
        self.score.orders().rhs
    }
}
