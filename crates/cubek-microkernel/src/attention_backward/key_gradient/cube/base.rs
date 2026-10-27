use crate::attention::{G, Q, S};
use crate::attention_backward::key_gradient::plane::{
    PlaneKeyGradientMicrokernel, PlaneKeyGradientMicrokernelSettings,
};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Axis, Level, Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A box of keys of one key head a cube owns: its planes each take a box of them with the plane
/// microkernel's step, the cube staging its keys and values once and, for each head of the group,
/// each stage of queries and output gradient once for all its planes.
///
/// The cube owns the staging, its ring met on `sync_cube`; each plane holds its windows and its
/// sums, met on `sync_plane`, and writes its own keys' and values' gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeKeyGradientMicrokernel {
    pub plane: PlaneKeyGradientMicrokernel,
    pub settings: CubeKeyGradientMicrokernelSettings,
}

/// The plane microkernel's settings and the planes along a cube's keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeKeyGradientMicrokernelSettings {
    pub plane: PlaneKeyGradientMicrokernelSettings,
    /// Planes along the keys, which read the queries it stages.
    pub planes_along_keys: NonZeroUsize,
}

impl CubeKeyGradientMicrokernelSettings {
    /// The plane microkernel's default, two planes along the keys.
    pub const DEFAULT: Self = Self {
        plane: PlaneKeyGradientMicrokernelSettings::DEFAULT,
        planes_along_keys: NonZeroUsize::new(2).unwrap(),
    };
}

impl core::fmt::Display for CubeKeyGradientMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes_kv={}", self.plane, self.planes_along_keys)
    }
}

impl CubeKeyGradientMicrokernel {
    /// The levels inside one cube's keys, which a routine states the rest of the partitioning
    /// around: its planes along the keys, each on the plane microkernel's stage
    /// ([`PlaneKeyGradientMicrokernel::step_tiling`]), walked through all of `Q`, and that walk
    /// through every head of the group.
    pub fn tiling(&self) -> Levels {
        self.plane
            .step_tiling()
            .planes(&self.planes())
            .walk_every(&[Q])
            .walk_every(&[G])
    }

    /// Keys a cube's box holds.
    pub fn seq_kv(&self) -> usize {
        self.plane.seq_kv() * self.settings.planes_along_keys.get()
    }

    /// Queries one stage holds.
    pub fn seq_q(&self) -> usize {
        self.plane.seq_q()
    }

    /// Planes a cube holds.
    pub fn plane_count(&self) -> usize {
        self.settings.planes_along_keys.get()
    }

    /// The score's grid below the cube's planes.
    pub fn score_levels(&self) -> Vec<Level> {
        self.plane.score_levels(&self.planes())
    }

    /// The grid of the gradient against the probabilities below the cube's planes.
    pub fn score_gradient_levels(&self) -> Vec<Level> {
        self.plane.score_gradient_levels(&self.planes())
    }

    /// Bytes of shared memory one cube holds beside its stages: its keys and values, staged once,
    /// and every plane's own while it walks.
    pub fn window_bytes(&self, item: &AttentionBackwardWorkItem) -> usize {
        let staged = self.seq_kv() * (item.head_dim + item.val_dim) * item.element.size();
        staged + self.plane_count() * self.plane.walk_bytes(item)
    }

    /// Its planes along the keys.
    fn planes(&self) -> [(Axis, usize); 1] {
        [(S, self.settings.planes_along_keys.get())]
    }
}

/// Its planes' microkernel and its own settings.
impl core::fmt::Display for CubeKeyGradientMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} planes_kv={}",
            self.plane, self.settings.planes_along_keys
        )
    }
}

impl Microkernel for CubeKeyGradientMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = CubeKeyGradientMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type Partitioning = Partitioning;

    const DECLINE: &'static str = PlaneKeyGradientMicrokernel::DECLINE;

    /// The default, then one plane and four planes a cube, then each plane setting beyond the
    /// default.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        let lead = Self::Settings::DEFAULT;
        let cubes = [1, 4].map(|planes| Self::Settings {
            planes_along_keys: NonZeroUsize::new(planes).unwrap(),
            ..lead
        });
        let planes = PlaneKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|plane| Self::Settings { plane, ..lead });
        core::iter::once(lead).chain(cubes).chain(planes).collect()
    }

    /// The plane microkernel's candidates, a plane each along the cube's keys.
    fn candidates(
        hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let cube = *settings;
        PlaneKeyGradientMicrokernel::candidates(hw, item, &settings.plane).map(move |plane| Self {
            plane,
            settings: cube,
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
