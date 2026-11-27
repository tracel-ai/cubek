use crate::attention::plane::ScoreHolder;
use crate::attention::{D, DV, G, Q, S};
use crate::attention_backward::{
    AttentionBackwardOperand, AttentionBackwardWorkItem, BoundProduct, estimate_beside,
};
use crate::matmul::plane::{PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings};
use crate::matmul::{MatmulDim, MatmulWorkItem};
use core::num::NonZeroUsize;
use cubecl::cmma::MatrixLayout;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Dimension, FragmentSource, Level, Levels, Partitioning, Scratch};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// The keys of one key head a plane owns: for each head of the group, a stage of queries at a
/// time, their scores keys first in the plane's window, the probabilities read back there from
/// each query's logsumexp and cast for the fragments to load, the gradient against them in another
/// window, cast in turn; the values' and the keys' gradients in fragments held across every stage
/// of every head.
///
/// Its four multiplications are the shared plane fragment microkernel's, each on the instruction
/// the tile picks for the plane's tile of it, which it cuts into calls and walks:
/// the score, `key · queryᵀ`, its keys by the stage's queries walking the head dim; the gradient
/// against the probabilities, `value · out_gradientᵀ`, the same grid walking the value dim; the
/// values' gradient, `probabilities · out_gradient`, its keys by the value dim walking the stage's
/// queries; and the keys' gradient, `score_gradient · query`, its keys by the head dim walking
/// them. The keys' gradient is the walk's; the other three are contracted on their own grids.
/// Each of the plane's products is its tile in elements ([`PlaneUnit::Elements`]).
///
/// [`PlaneUnit::Elements`]: crate::matmul::plane::PlaneUnit::Elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneKeyGradientMicrokernel {
    pub settings: PlaneKeyGradientMicrokernelSettings,
    /// The score's fragments: its keys by a stage's queries, walking the whole head dim.
    pub score: PlaneFragmentMicrokernel,
    /// The fragments of the gradient against the probabilities: the score's grid, walking the
    /// whole value dim.
    pub score_gradient: PlaneFragmentMicrokernel,
    /// The values' gradient's fragments: its keys by the whole value dim, walking a stage's
    /// queries.
    pub value_gradient: PlaneFragmentMicrokernel,
    /// The keys' gradient's fragments: its keys by the whole head dim, walking a stage's queries.
    pub key_gradient: PlaneFragmentMicrokernel,
}

/// The instructions a plane's tile holds along the keys and along a stage of queries, how much of
/// the keys' gradient its scale bounces through shared memory at once, and where it keeps the
/// score ([`ScoreHolder`]): in windows of shared memory, read through cmma, or in its own manual
/// `mma` fragments. The instruction is not a setting: it follows from where the score is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneKeyGradientMicrokernelSettings {
    pub seq_kv: NonZeroUsize,
    pub seq_q: NonZeroUsize,
    pub scratch: Scratch,
    pub score: ScoreHolder,
}

impl PlaneKeyGradientMicrokernelSettings {
    pub const DEFAULT: Self = Self::fragments(2, 2);

    /// `keys` units of keys against `queries` units of queries, one fragment bounced at a time.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn fragments(keys: usize, queries: usize) -> Self {
        Self {
            seq_kv: NonZeroUsize::new(keys).unwrap(),
            seq_q: NonZeroUsize::new(queries).unwrap(),
            scratch: Scratch::OneTile,
            score: ScoreHolder::Window,
        }
    }

    /// These settings on the manual `mma`, the score kept in its fragments.
    pub const fn score_in_fragments(self) -> Self {
        Self {
            score: ScoreHolder::Fragments,
            ..self
        }
    }
}

impl core::fmt::Display for PlaneKeyGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fragments={}x{}", self.seq_kv, self.seq_q)?;
        if self.score == ScoreHolder::Fragments {
            write!(f, " held")?;
        }
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

    /// The levels inside one stage of one plane's keys: the plane's tile of both gradients, its
    /// keys by one head's queries by the head and value dims whole, which the tile cuts into each
    /// product's calls and walks. The shared microkernel steps the level below a stage, so the
    /// tile states one step there, and each product is one call a stage. A cube places its planes
    /// above them.
    pub fn step_tiling(&self) -> Levels {
        Levels::leaf(&[
            (S, self.seq_kv()),
            (G, 1),
            (Q, self.seq_q()),
            (D, self.key_gradient.tile_extent(D)),
            (DV, self.value_gradient.tile_extent(DV)),
        ])
        .walk(&[(Q, 1)])
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
    pub fn score_levels(&self, planes: &[(Dimension, usize)]) -> Vec<Level> {
        self.score.tile_levels(planes)
    }

    /// The grid of the gradient against the probabilities below the walker's level of planes.
    pub fn score_gradient_levels(&self, planes: &[(Dimension, usize)]) -> Vec<Level> {
        self.score_gradient.tile_levels(planes)
    }

    /// Where a plane keeps a step's score and the gradient against its probabilities: its
    /// settings', which its instruction was picked for.
    pub fn holder(&self) -> ScoreHolder {
        self.settings.score
    }

    /// The scratch the gradients' sums bounce through as their scale is taken: the settings', for
    /// a sum that drains by bouncing through shared memory, a cmma one; none for one whose units
    /// hold their own cells and scale them in place.
    pub fn sum_scratch(&self) -> Scratch {
        match self.key_gradient.cost.bounced_cell {
            Some(_) => self.settings.scratch,
            None => Scratch::None,
        }
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
        let windows = match self.holder() {
            ScoreHolder::Window => {
                self.seq_kv() * self.seq_q() * (2 * item.sum.size() + item.element.size())
            }
            ScoreHolder::Fragments => 0,
        };
        let bounced = self
            .key_gradient
            .scratch_bytes(&item.key_gradient(item.element), self.sum_scratch())
            + self
                .value_gradient
                .scratch_bytes(&item.value_gradient(item.element), self.sum_scratch());
        windows + bounced
    }

    /// The plane over the tile's picks for its four products, `settings` counting the device's
    /// matrix unit ([`candidates`](Microkernel::candidates)). `None` where the tile picks no
    /// instruction for one of them.
    fn picked(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: PlaneKeyGradientMicrokernelSettings,
    ) -> Option<Self> {
        let grid = PlaneFragmentMicrokernelSettings {
            along_m: settings.seq_kv,
            along_n: settings.seq_q,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        };
        let (score_item, gradient_item) = (item.score(S), item.score_gradient(S));
        let value_item = item.value_gradient(item.element);
        let key_item = item.key_gradient(item.element);
        let (head, value) = (item.head_dim, item.val_dim);
        let [keys, queries] = PlaneFragmentMicrokernel::unit_tile(hw, &score_item, &grid)?;
        let (score_tile, gradient_tile) = ([keys, queries, head], [keys, queries, value]);
        let (value_tile, key_tile) = ([keys, value, queries], [keys, head, queries]);
        let holder = settings.score;
        let score = holder.sum_plane(hw, &score_item, grid, score_tile)?;
        let score_gradient = holder.sum_plane(hw, &gradient_item, grid, gradient_tile)?;
        Some(Self {
            settings,
            value_gradient: holder.reader_plane(
                hw,
                &score,
                &value_item,
                FragmentSource::Values(value_item.elements.rhs),
                grid,
                value_tile,
            )?,
            key_gradient: holder.reader_plane(
                hw,
                &score_gradient,
                &key_item,
                FragmentSource::Values(key_item.elements.rhs),
                grid,
                key_tile,
            )?,
            score,
            score_gradient,
        })
    }

    /// `product` on the score's instruction, `along_m` by `along_n` fragments, `depth` steps of its
    /// contraction: a product that states its fragments. The hypercube's share of the query's
    /// gradient states its own this way: its planes split a stage's part in a grid of whole
    /// fragments, which a plane's tile alone does not say.
    pub(crate) fn beside(
        hw: &Hardware,
        score: PlaneFragmentMicrokernel,
        product: &MatmulWorkItem,
        (along_m, along_n, depth): (NonZeroUsize, NonZeroUsize, NonZeroUsize),
    ) -> PlaneFragmentMicrokernel {
        PlaneFragmentMicrokernel {
            instruction: score.instruction,
            step: None,
            cost: score.instruction.cost(&hw.target),
            orders: score.instruction.orders(product.layouts),
            settings: PlaneFragmentMicrokernelSettings {
                along_m,
                along_n,
                depth,
                ..score.settings
            },
            dimensions: product.dimensions,
            blocks: None,
            k_stated: crate::matmul::plane::PlaneFragmentKStatement::Caller,
            unit: crate::matmul::plane::PlaneUnit::Instruction,
        }
    }

    /// `extent` in steps of `step`, where they divide it: the hypercube share's fragments of the
    /// part its planes split ([`beside`](Self::beside)).
    pub(crate) fn steps(extent: usize, step: usize) -> Option<NonZeroUsize> {
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
    type CubeSpace = Partitioning;

    const DECLINE: &'static str = "the plane backward microkernel runs on a cmma instruction, or \
         a manual mma of 16-bit operands whose accumulator the device reads as its A, of the \
         operands' element summing in f32, whose k divides the head and value dims and a stage's \
         queries and whose n divides the head and value dims, on a device that commits to one \
         plane width";

    /// The default, a fragment of each, more fragments along the keys or along the queries, and
    /// the whole grid of both gradients bounced at once; then the default and the wider grids
    /// with the score held in its fragments, and a fragment of keys against two and four of
    /// queries: a plane's gradients over a wide head dim held without spilling its registers.
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
            Self::Settings::DEFAULT.score_in_fragments(),
            Self::Settings::fragments(4, 2).score_in_fragments(),
            Self::Settings::fragments(2, 4).score_in_fragments(),
            Self::Settings::fragments(1, 2).score_in_fragments(),
            Self::Settings::fragments(1, 4).score_in_fragments(),
        ]
    }

    /// The tile's picks for the plane's four products, held where the settings say: the score,
    /// its keys by a stage's queries by the whole head dim; the gradient against the
    /// probabilities, the same keys and queries by the whole value dim; the values' gradient and
    /// the keys' gradient, its keys by the whole value and head dims, contracting the stage's
    /// queries, reading the probabilities and the score's gradient. The keys and queries are the
    /// settings' counts of the device's matrix unit. Each is asked of the tile with the sources its
    /// kernel opens the sum on: in windows, two drained sums and two of their cast windows; held,
    /// two sums contracted again, each gradient reading runs of its sum's cells. None where the
    /// tile picks no instruction for one of them.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        Self::picked(hw, item, *settings).into_iter()
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
        for product in [
            BoundProduct {
                microkernel: self.score,
                lhs: Some(Operand::Key),
                rhs: Some(Operand::Query),
                out: None,
            },
            BoundProduct {
                microkernel: self.score_gradient,
                lhs: Some(Operand::Value),
                rhs: Some(Operand::OutGradient),
                out: None,
            },
            BoundProduct {
                microkernel: self.value_gradient,
                lhs: None,
                rhs: Some(Operand::OutGradient),
                out: Some(Operand::ValueGradient),
            },
            BoundProduct {
                microkernel: self.key_gradient,
                lhs: None,
                rhs: Some(Operand::Query),
                out: Some(Operand::KeyGradient),
            },
        ] {
            product.restate(vectorization);
        }
        for (operand, product) in [
            (Operand::KeyGradient, self.key_gradient),
            (Operand::ValueGradient, self.value_gradient),
        ] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: product.leaf_extent(MatmulDim::N),
            });
            // Its sum drains in the lines the tile writes it in, along the gradient's rows.
            vectorization.require(
                product
                    .cost
                    .line_requirement(operand, MatrixLayout::RowMajor),
            );
        }
        for operand in [Operand::Logsumexp, Operand::Delta] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }

    /// Each multiplication's shared microkernel sets the work of its fragments; the plane's units
    /// hold the four at once. Its stages are its routine's to count.
    fn estimate(
        &self,
        item: &AttentionBackwardWorkItem,
        _partitioning: &Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate_beside(
            [
                (self.score, item.score(S)),
                (self.score_gradient, item.score_gradient(S)),
                (self.value_gradient, item.value_gradient(item.element)),
                (self.key_gradient, item.key_gradient(item.element)),
            ],
            planes,
            estimate,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `f16` summed in `f32`, `m × n × k`.
    fn shape((m, n, k): (u32, u32, u32)) -> MmaConfig {
        MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m,
            n,
            k,
        }
    }

    /// An H100 offering the fragment API's three `f16` shapes and the manual `16 × 8 × 16`.
    fn nvidia() -> Hardware {
        let hw = devices::offering_cmma(
            devices::h100(),
            [(16, 16, 16), (32, 8, 16), (8, 32, 16)].map(shape),
        );
        devices::offering_mma(hw, [shape((16, 8, 16))])
    }

    /// An M2 offering `simdgroup_matrix`'s `8 × 8 × 8` through both transports.
    fn m2() -> Hardware {
        let eight = shape((8, 8, 8));
        devices::offering_mma(devices::offering_cmma(devices::m2_pro(), [eight]), [eight])
    }

    fn item() -> AttentionBackwardWorkItem {
        AttentionBackwardWorkItem {
            element: F16,
            sum: F32,
            group: 1,
            head_dim: 64,
            val_dim: 64,
        }
    }

    /// On NVIDIA every setting the plane lists finds the tile's picks, the held ones keeping
    /// both sums in their registers; on an M2, whose one `8 × 8 × 8` both transports offer, the
    /// plane reads its sums through windows only.
    #[test]
    fn the_held_planes_run_where_a_sum_stays_in_its_registers() {
        let (nvidia, m2) = (nvidia(), m2());
        for settings in PlaneKeyGradientMicrokernel::settings(&nvidia) {
            let planes: Vec<_> =
                PlaneKeyGradientMicrokernel::candidates(&nvidia, &item(), &settings).collect();
            assert!(!planes.is_empty(), "{settings} on NVIDIA finds no plane");
            let held = settings.score == ScoreHolder::Fragments;
            let on_m2 = PlaneKeyGradientMicrokernel::candidates(&m2, &item(), &settings).count();
            assert_eq!(on_m2 == 0, held, "{settings} on an M2");
        }
    }
}
