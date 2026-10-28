use crate::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use crate::attention_backward::key_gradient::plane::PlaneKeyGradientMicrokernel;
use crate::attention_backward::{
    AttentionBackwardOperand, AttentionBackwardWorkItem, BoundProduct, estimate_beside,
};
use crate::matmul::plane::PlaneFragmentMicrokernel;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Level, Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Every cube of the launch owning a box of keys of one key head, as the cube microkernel does,
/// and adding its share of the query's gradient into one sum every cube that sees those queries
/// adds into: that sum is the hypercube's dependent work.
///
/// Each plane takes every stage with the plane microkernel's step, then contracts
/// `scale · score_gradient · key` for the stage's queries out of the score's gradient the step
/// leaves cast in its window. The planes, which split the stage's keys, meet in shared memory, and
/// the cube adds their sum into the query's gradient atomically, once a stage
/// ([`QueryGradientShare`](crate::attention_backward::key_gradient::cube::QueryGradientShare)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeKeyGradientMicrokernel {
    pub cube: CubeKeyGradientMicrokernel,
    /// The query gradient's fragments: a stage's queries by the whole head dim, walking the
    /// plane's keys.
    pub query_gradient: PlaneFragmentMicrokernel,
    pub settings: HypercubeKeyGradientMicrokernelSettings,
}

/// The cube microkernel's settings every cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeKeyGradientMicrokernelSettings {
    pub cube: CubeKeyGradientMicrokernelSettings,
}

impl HypercubeKeyGradientMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        cube: CubeKeyGradientMicrokernelSettings::DEFAULT,
    };
}

impl core::fmt::Display for HypercubeKeyGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.cube)
    }
}

impl HypercubeKeyGradientMicrokernel {
    /// The cube microkernel's levels: a cube's planes along its keys, walked through all of `Q`
    /// and every head of the group.
    pub fn tiling(&self) -> Levels {
        self.cube.tiling()
    }

    /// Keys a cube's box holds.
    pub fn seq_kv(&self) -> usize {
        self.cube.seq_kv()
    }

    /// Queries one stage holds.
    pub fn seq_q(&self) -> usize {
        self.cube.seq_q()
    }

    /// Planes a cube holds.
    pub fn plane_count(&self) -> usize {
        self.cube.plane_count()
    }

    /// The score's grid below the cube's planes.
    pub fn score_levels(&self) -> Vec<Level> {
        self.cube.score_levels()
    }

    /// The grid of the gradient against the probabilities below the cube's planes.
    pub fn score_gradient_levels(&self) -> Vec<Level> {
        self.cube.score_gradient_levels()
    }

    /// Bytes of shared memory one cube holds beside its stages: the cube microkernel's, the
    /// fragments every plane's part of the query's gradient bounces through, and, where planes
    /// split the keys, the stage's query gradient they meet in.
    pub fn window_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let bounced = self.query_gradient.scratch_bytes(
            &item.summed_query_gradient(item.element),
            self.cube.plane.settings.scratch,
        );
        let met = match self.plane_count() {
            1 => 0,
            _ => self.seq_q() * item.head_dim * item.sum.size(),
        };
        self.cube.window_bytes(item) + self.plane_count() * bounced + met
    }
}

/// Its cube's microkernel.
impl core::fmt::Display for HypercubeKeyGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.cube)
    }
}

impl Microkernel for HypercubeKeyGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = HypercubeKeyGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type CubeSpace = Partitioning;

    const DECLINE: &'static str = PlaneKeyGradientMicrokernel::DECLINE;

    /// The cube microkernel's.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .map(|cube| Self::Settings { cube })
            .collect()
    }

    /// The cube microkernel's candidates, the query's gradient on their score's instruction: a
    /// stage's queries by the head dim, walking the plane's keys in steps of its `k`.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let hypercube = *settings;
        let query_item = item.summed_query_gradient(item.element);
        let head = item.head_dim;
        let candidates: Vec<_> = CubeKeyGradientMicrokernel::candidates(hw, item, &settings.cube)
            .filter_map(|cube| {
                let plane = cube.plane;
                let [_, (_, n), (_, k)] = plane.score.instruction.shape();
                let head_columns = PlaneKeyGradientMicrokernel::steps(head, n)?;
                let key_steps =
                    PlaneKeyGradientMicrokernel::steps(plane.settings.seq_kv.get() * n, k)?;
                Some(Self {
                    cube,
                    query_gradient: PlaneKeyGradientMicrokernel::beside(
                        plane.score,
                        &query_item,
                        (plane.settings.seq_q, head_columns, key_steps),
                    ),
                    settings: hypercube,
                })
            })
            .collect();
        candidates.into_iter()
    }

    /// Its cube's, and the query's gradient's: it reads the keys, and adds into the sum a value at
    /// a time, as an atomic element is added.
    fn vectorize(
        &self,
        partitioning: &Partitioning,
        vectorization: &mut Vectorization<AttentionBackwardOperand>,
    ) {
        use AttentionBackwardOperand as Operand;
        self.cube.vectorize(partitioning, vectorization);
        BoundProduct {
            microkernel: self.query_gradient,
            lhs: None,
            rhs: Some(Operand::Key),
            out: Some(Operand::QueryGradient),
        }
        .restate(vectorization);
        vectorization.require(VectorRequirement::Elementwise {
            operand: Operand::QueryGradient,
        });
    }

    /// Its cube's work, and the query's gradient's beside it on the same instruction.
    fn estimate(
        &self,
        item: &AttentionBackwardWorkItem,
        partitioning: &Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.cube.estimate(item, partitioning, planes, estimate);
        let cube = estimate.work.take();
        estimate_beside(
            [(
                self.query_gradient,
                item.summed_query_gradient(item.element),
            )],
            planes,
            estimate,
        );
        estimate.work = cube.and_then(|cube| estimate.work.take().map(|block| cube.beside(block)));
    }
}
