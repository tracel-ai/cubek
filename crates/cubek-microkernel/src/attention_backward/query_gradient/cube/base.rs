use crate::attention::{G, Q, S};
use crate::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, PlaneQueryGradientMicrokernelSettings,
};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Axis, Level, Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A box of queries of some heads of the group a cube owns: its planes each take a box of one
/// head's queries with the plane microkernel's step, the cube staging every query's own operands
/// once and each stage of keys and values once for all its planes. Every head of the group reads
/// the same keys, which is what makes a cube worth more than its planes.
///
/// The cube owns the staging, its ring met on `sync_cube`; each plane holds its windows and its
/// sum, met on `sync_plane`, and writes its own queries' gradient and dot products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeQueryGradientMicrokernel {
    pub plane: PlaneQueryGradientMicrokernel,
    pub settings: CubeQueryGradientMicrokernelSettings,
    /// Heads of the group a cube holds: as many as the settings ask, and no more than the group
    /// has.
    pub heads: usize,
}

/// The plane microkernel's settings, the heads a cube holds and the planes along each head's
/// queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeQueryGradientMicrokernelSettings {
    pub plane: PlaneQueryGradientMicrokernelSettings,
    /// Heads of the group one cube holds, which read the keys it stages.
    pub heads: NonZeroUsize,
    /// Planes along one head's queries.
    pub planes_along_queries: NonZeroUsize,
}

impl CubeQueryGradientMicrokernelSettings {
    /// The plane microkernel's default, two planes along one head's queries.
    pub const DEFAULT: Self = Self {
        plane: PlaneQueryGradientMicrokernelSettings::DEFAULT,
        heads: NonZeroUsize::new(1).unwrap(),
        planes_along_queries: NonZeroUsize::new(2).unwrap(),
    };
}

impl core::fmt::Display for CubeQueryGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} heads={} planes_q={}",
            self.plane, self.heads, self.planes_along_queries
        )
    }
}

impl CubeQueryGradientMicrokernel {
    /// The levels inside one cube's queries, which a routine states the rest of the partitioning
    /// around: its planes over the heads and the queries, each on the plane microkernel's stage
    /// ([`PlaneQueryGradientMicrokernel::step_tiling`]), walked through all of `S`.
    pub fn tiling(&self) -> Levels {
        self.plane
            .step_tiling()
            .planes(&self.planes())
            .walk_every(&[S])
    }

    /// Queries of one head a cube's box holds.
    pub fn seq_q(&self) -> usize {
        self.plane.seq_q() * self.settings.planes_along_queries.get()
    }

    /// Keys one stage holds.
    pub fn seq_kv(&self) -> usize {
        self.plane.seq_kv()
    }

    /// Planes a cube holds.
    pub fn plane_count(&self) -> usize {
        self.heads * self.settings.planes_along_queries.get()
    }

    /// The score's grid below the cube's planes.
    pub fn score_levels(&self) -> Vec<Level> {
        self.plane.score_levels(&self.planes())
    }

    /// The grid of the gradient against the probabilities below the cube's planes.
    pub fn score_gradient_levels(&self) -> Vec<Level> {
        self.plane.score_gradient_levels(&self.planes())
    }

    /// Bytes of shared memory one cube holds beside its stages: its box of the query and of the
    /// output's gradient, staged once, and every plane's own while it walks.
    pub fn window_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let staged =
            self.heads * self.seq_q() * (item.head_dim + item.val_dim) * item.element.size();
        staged + self.plane_count() * self.plane.walk_bytes(item)
    }

    /// Its planes along the heads and the queries.
    fn planes(&self) -> [(Axis, usize); 2] {
        [
            (G, self.heads),
            (Q, self.settings.planes_along_queries.get()),
        ]
    }
}

/// Its planes' microkernel and its own settings.
impl core::fmt::Display for CubeQueryGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} heads={} planes_q={}",
            self.plane, self.heads, self.settings.planes_along_queries
        )
    }
}

impl Microkernel for CubeQueryGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = CubeQueryGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type Partitioning = Partitioning;

    const DECLINE: &'static str = PlaneQueryGradientMicrokernel::DECLINE;

    /// The default, then one plane, four planes, and two heads a cube, then each plane setting
    /// beyond the default.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        let lead = Self::Settings::DEFAULT;
        let cubes = [(1, 1), (1, 4), (2, 1)].map(|(heads, planes)| Self::Settings {
            heads: NonZeroUsize::new(heads).unwrap(),
            planes_along_queries: NonZeroUsize::new(planes).unwrap(),
            ..lead
        });
        let planes = PlaneQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|plane| Self::Settings { plane, ..lead });
        core::iter::once(lead).chain(cubes).chain(planes).collect()
    }

    /// The plane microkernel's candidates, a plane each for every head the cube holds along every
    /// plane of their queries.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let (cube, heads) = (*settings, settings.heads.get().min(item.group));
        PlaneQueryGradientMicrokernel::candidates(hw, item, &settings.plane).map(move |plane| {
            Self {
                plane,
                settings: cube,
                heads,
            }
        })
    }

    /// Its planes', whose reads its stages copy.
    fn vectorize(
        &self,
        partitioning: &Partitioning,
        vectorization: &mut Vectorization<AttentionBackwardOperand>,
    ) {
        self.plane.vectorize(partitioning, vectorization);
    }

    /// Its planes' work; its stages are its routine's to count.
    fn estimate(
        &self,
        item: &AttentionBackwardWorkItem,
        partitioning: &Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.plane.estimate(item, partitioning, planes, estimate);
    }
}
