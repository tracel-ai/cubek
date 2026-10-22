use crate::attention::cube::{CubeAttentionMicrokernelSettings, CubeAttentionPlaneSettings};
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, AttentionPartitioning,
    AttentionSlotShape, AttentionStagingCount, AttentionWorkItem, CubePlaneKeys, CubePlaneLayout,
    CubePlaneMicrokernel, CubePlaneReads, G, Q, S,
};
use crate::matmul::MatmulOrder;
use crate::staging::StageRing;
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Levels, Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A box of queries of some heads of the group a cube owns: its planes each take the rows of one
/// head with the plane microkernel `M`. Where `M` reads a stage, the cube stages the keys and
/// values once for all its planes — what makes a cube worth more than its planes, since every
/// head of the group reads the same keys; where `M` reads keys where they lie, each plane reads
/// its own.
///
/// The cube owns the staging, its ring met on `sync_cube` and laid out as its settings say; each
/// plane takes every step into its rows, in windows of its own. Where planes split the keys, each
/// takes its block of every step, and the planes holding the same rows merge their partials before
/// the cube writes them; otherwise each plane writes its own rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeAttentionMicrokernel<M: CubePlaneMicrokernel> {
    pub plane: M,
    pub settings: CubeAttentionMicrokernelSettings<M::Settings>,
    /// Heads of the group a cube holds, a plane each: as many as the settings ask, and no more
    /// than the group has.
    pub heads: usize,
    /// The settings' ring as this device holds its stages of keys and values
    /// ([`StageRing::resolve`]).
    pub ring: StageRing,
    /// Units of each plane, the device's plane width: what a merged row's work is shared among
    /// where planes split the same rows' keys.
    pub units: usize,
    /// Steps of keys a page of the cache holds, where the cube walks its keys a page at a time
    /// ([`over_pages`](Self::over_pages)).
    pub page_steps: Option<NonZeroUsize>,
}

impl<M: CubePlaneMicrokernel> CubeAttentionMicrokernel<M> {
    /// This cube walking `item`'s paged cache a page at a time where its planes read keys where
    /// they lie and a page holds a whole number of its steps, more than one; otherwise still
    /// walking steps.
    pub(crate) fn over_pages(self, item: &AttentionWorkItem) -> Self {
        let keys = self.keys();
        let page_steps = match (M::READS, item.page_size) {
            (CubePlaneReads::Global, Some(page_size))
                if page_size > keys && page_size.is_multiple_of(keys) =>
            {
                NonZeroUsize::new(page_size / keys)
            }
            _ => None,
        };
        Self { page_steps, ..self }
    }

    /// Keys one region of the cube's walk covers: a step, or a page where it walks over pages.
    pub fn region_keys(&self) -> usize {
        self.keys() * self.page_steps.map_or(1, NonZeroUsize::get)
    }

    /// The levels inside one step of the cube's walk through `S`, which its partitioning states
    /// ([`AttentionPartitioning::new`]): its heads and the planes along their queries and keys,
    /// each running the plane microkernel's box.
    pub fn tiling(&self) -> Levels {
        self.plane.tiling().planes(&[
            (G, self.heads),
            (Q, self.settings.planes_along_queries.get()),
            (S, self.settings.planes_along_keys.get()),
        ])
    }

    /// Query positions of one head a cube's box holds.
    pub fn queries(&self) -> usize {
        self.plane.queries() * self.settings.planes_along_queries.get()
    }

    /// Keys one stage of the ring holds: a block for every plane along them.
    pub fn keys(&self) -> usize {
        self.plane.keys() * self.settings.planes_along_keys.get()
    }

    /// How the slots its planes that split the same rows' keys leave their rows in lie, `value`
    /// elements a row: a slot for each of those planes, a merged row's work shared among a
    /// plane's units.
    pub fn merge_shape(&self, value: usize) -> AttentionSlotShape {
        AttentionSlotShape {
            units: self.units,
            value,
            splits: self.settings.planes_along_keys.get(),
        }
    }

    /// How it lays its planes over its box.
    pub fn layout(&self) -> CubePlaneLayout {
        CubePlaneLayout {
            heads: self.heads,
            planes_along_queries: self.settings.planes_along_queries.get(),
            planes_along_keys: self.settings.planes_along_keys.get(),
        }
    }

    /// The storage a stage of keys and values over `stage` is held in: as the settings lay it
    /// out, blocked as its planes read it.
    pub fn stage_storage(&self, stage: &Space) -> StageStorage {
        self.plane.stage_storage(self.settings.layout, stage)
    }

    /// Bytes of shared memory one cube holds alone on `hw`, its stages held as `storage`: what it
    /// walks with ([`walk_shared_memory`](Self::walk_shared_memory)), and where its planes that
    /// split the same rows' keys meet.
    pub fn shared_memory(
        &self,
        hw: &Hardware,
        item: &AttentionWorkItem,
        storage: &StageStorage,
    ) -> usize {
        self.walk_shared_memory(hw, item, storage) + self.plane.merge_bytes(item, self.layout())
    }

    /// Bytes of shared memory one cube holds on `hw` while it walks: every plane's windows, and,
    /// where its planes read a stage, its box of the query once and each stage in flight of keys
    /// and values, held as `storage`.
    pub fn walk_shared_memory(
        &self,
        hw: &Hardware,
        item: &AttentionWorkItem,
        storage: &StageStorage,
    ) -> usize {
        let planes = self.layout().planes() * self.plane.window_bytes(item);
        let staged = match M::READS {
            CubePlaneReads::Stage => {
                let query = self.heads * self.queries() * item.head * item.elements.query.size();
                let stage: usize = item
                    .stage(self.keys(), self.key_order())
                    .iter()
                    .map(|operand| operand.bytes(hw, storage))
                    .sum();
                query + self.ring.bytes(stage)
            }
            CubePlaneReads::Global => 0,
        };
        planes + staged
    }

    /// What one cube reads of each operand: its box of the query and every key and value of its
    /// walk, once for all its planes where it stages them, once for every group of planes holding
    /// different rows where each plane reads its own.
    fn staging(&self, item: &AttentionWorkItem) -> AttentionStagingCount {
        let (key, stages_in_flight) = match M::READS {
            CubePlaneReads::Stage => (1, self.ring.stages_in_flight.get() as u32),
            CubePlaneReads::Global => (self.layout().row_groups() as u64, 1),
        };
        AttentionStagingCount {
            keys: self.keys(),
            stages_in_flight,
            runs: AttentionOperands {
                query: item.head,
                key: item.head,
                value: item.value,
                out: item.value,
            },
            readers: AttentionOperands {
                query: 1,
                key,
                value: key,
                out: 1,
            },
        }
    }
}

/// Its planes and ring, and the steps a page holds where it walks pages.
impl<M: CubePlaneMicrokernel> core::fmt::Display for CubeAttentionMicrokernel<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.plane, self.settings)?;
        if let Some(steps) = self.page_steps {
            write!(f, " page_steps={steps}")?;
        }
        Ok(())
    }
}

impl<M: CubePlaneMicrokernel> Microkernel for CubeAttentionMicrokernel<M> {
    type WorkItem = AttentionWorkItem;
    type Settings = CubeAttentionMicrokernelSettings<M::Settings>;
    type Operand = AttentionOperand;
    type Partitioning = AttentionPartitioning;

    const DECLINE: &'static str = M::DECLINE;

    /// The cube's settings over planes of its kind
    /// ([`CubeAttentionPlaneSettings::cube_settings`]).
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        M::Settings::cube_settings(hw)
    }

    /// The plane microkernel's candidates, a plane each for every head the cube holds along every
    /// plane of its queries and of their keys, the ring resolved against the stages the cube's
    /// units fill; planes split the same rows' keys only where the plane microkernel merges them.
    fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let (heads, cube) = (settings.heads.get().min(item.group), *settings);
        let splits = settings.planes_along_keys.get() > 1;
        M::candidates(hw, item, &settings.plane)
            .filter(move |_| M::KEYS == CubePlaneKeys::Split || !splits)
            .filter_map(move |plane| {
                let requested = Self {
                    plane,
                    settings: cube,
                    heads,
                    ring: cube.ring,
                    units: hw.plane_dim().ok()? as usize,
                    page_steps: None,
                }
                .over_pages(item);
                let ring = cube.ring.resolve(
                    hw,
                    &requested
                        .served(item)
                        .stage(requested.keys(), requested.key_order()),
                    requested.layout().planes() * requested.units,
                );
                Some(Self { ring, ..requested })
            })
    }

    /// Its planes', whose reads its ring copies where it stages them.
    fn vectorize(
        &self,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        self.plane.vectorize(partitioning, vectorization);
    }

    /// Its plane microkernel's work, and the staging, which it places: its box of the query, and
    /// every key and value of the walk in runs of a row each, as many times over as it has
    /// readers.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        partitioning: &AttentionPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.plane.estimate(item, partitioning, planes, estimate);
        self.staging(item).count(item, partitioning, estimate);
    }
}

impl<M: CubePlaneMicrokernel> AttentionMicrokernel for CubeAttentionMicrokernel<M> {
    /// Its planes' microkernel's.
    fn key_order(&self) -> MatmulOrder {
        self.plane.key_order()
    }

    /// Its planes' microkernel's.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        self.plane.served(item)
    }
}
