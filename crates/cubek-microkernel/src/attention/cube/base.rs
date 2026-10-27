use crate::attention::cube::{
    CubeAttentionMicrokernelSettings, CubeAttentionPlane, CubePlaneKeys, CubePlaneLayout,
    CubePlaneReads,
};
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, AttentionStagingCount,
    AttentionWorkItem, G, Q, S,
};
use crate::split_keys_merge::SlotShape;
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Axis, Levels, Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::StageRing;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A box of queries of some heads of the group a cube owns: its planes each take the rows of one
/// head with its plane microkernel. Where its planes read a stage, the cube stages the keys and
/// values once for all of them — what makes a cube worth more than its planes, since every head
/// of the group reads the same keys; where they read keys where they lie, each plane reads its
/// own.
///
/// The cube owns the staging, its ring met on `sync_cube` and laid out as its settings say; each
/// plane takes every step into its rows, in windows of its own. Where planes split the keys, each
/// takes its block of every step, and the planes holding the same rows merge their partials before
/// the cube writes them; otherwise each plane writes its own rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeAttentionMicrokernel {
    pub plane: CubeAttentionPlane,
    pub settings: CubeAttentionMicrokernelSettings,
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
    /// (`over_pages`).
    pub page_steps: Option<NonZeroUsize>,
}

impl CubeAttentionMicrokernel {
    /// This cube walking `item`'s paged cache a page at a time where its planes read keys where
    /// they lie and a page holds a whole number of its steps, more than one; otherwise still
    /// walking steps.
    pub(crate) fn over_pages(self, item: &AttentionWorkItem) -> Self {
        let keys = self.seq_kv();
        let page_steps = match (self.plane.reads(), item.page_size) {
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
        self.seq_kv() * self.page_steps.map_or(1, NonZeroUsize::get)
    }

    /// The levels inside one step of the cube's walk through `S`, which its routine's placement
    /// states: its heads and the planes along their queries and keys,
    /// each running the plane microkernel's box.
    pub fn tiling(&self) -> Levels {
        self.plane.tiling().planes(&self.planes())
    }

    /// Its planes along each axis of its box: a plane for every head it holds, along every plane
    /// of a head's queries, along every plane of the same rows' keys.
    pub fn planes(&self) -> Vec<(Axis, usize)> {
        vec![
            (G, self.heads),
            (Q, self.settings.planes_along_queries.get()),
            (S, self.settings.planes_along_keys.get()),
        ]
    }

    /// Query positions of one head a cube's box holds.
    pub fn seq_q(&self) -> usize {
        self.plane.seq_q() * self.settings.planes_along_queries.get()
    }

    /// Keys one stage of the ring holds: a block for every plane along them.
    pub fn seq_kv(&self) -> usize {
        self.plane.seq_kv() * self.settings.planes_along_keys.get()
    }

    /// How the slots its planes that split the same rows' keys leave their rows in lie, `value`
    /// elements a row: a slot for each of those planes, a merged row's work shared among a
    /// plane's units.
    pub fn merge_shape(&self, value: usize) -> SlotShape {
        SlotShape {
            units: self.units,
            val_dim: value,
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
        let staged = match self.plane.reads() {
            CubePlaneReads::Stage => {
                let query = self.heads * self.seq_q() * item.head_dim * item.elements.query.size();
                let stage: usize = item
                    .stage(self.seq_kv(), self.key_order())
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
        let (key, stages_in_flight) = match self.plane.reads() {
            CubePlaneReads::Stage => (1, self.ring.stages_in_flight.get() as u32),
            CubePlaneReads::Global => (self.layout().row_groups() as u64, 1),
        };
        AttentionStagingCount {
            seq_kv: self.seq_kv(),
            stages_in_flight,
            runs: AttentionOperands {
                query: item.head_dim,
                key: item.head_dim,
                value: item.val_dim,
                out: item.val_dim,
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
impl core::fmt::Display for CubeAttentionMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.plane, self.settings)?;
        if let Some(steps) = self.page_steps {
            write!(f, " page_steps={steps}")?;
        }
        Ok(())
    }
}

impl Microkernel for CubeAttentionMicrokernel {
    type WorkItem = AttentionWorkItem;
    type Settings = CubeAttentionMicrokernelSettings;
    type Operand = AttentionOperand;
    type CubeSpace = Space;

    const DECLINE: &'static str = "the cube attention needs a plane microkernel the device runs";

    /// The cube's settings over each kind of plane, fragments, unit groups and shuffles, each
    /// list led by its kind's lead ([`CubeAttentionMicrokernelSettings::listed`]).
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [
            CubeAttentionMicrokernelSettings::FRAGMENT,
            CubeAttentionMicrokernelSettings::UNIT_GROUP,
            CubeAttentionMicrokernelSettings::SHUFFLE,
        ]
        .into_iter()
        .flat_map(CubeAttentionMicrokernelSettings::listed)
        .collect()
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
        CubeAttentionPlane::candidates(hw, item, &settings.plane)
            .into_iter()
            .filter(move |plane| plane.keys() == CubePlaneKeys::Split || !splits)
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
                        .stage(requested.seq_kv(), requested.key_order()),
                    requested.layout().planes() * requested.units,
                );
                Some(Self { ring, ..requested })
            })
    }

    /// Its plane microkernel's reason: what the device lacks for planes of that kind.
    fn decline(settings: &Self::Settings) -> &'static str {
        settings.plane.decline()
    }

    /// Its planes', whose reads its ring copies where it stages them.
    fn vectorize(&self, cube_space: &Space, vectorization: &mut Vectorization<AttentionOperand>) {
        self.plane.vectorize(cube_space, vectorization);
    }

    /// Its plane microkernel's work, and the staging, which it places: its box of the query, and
    /// every key and value of the walk in runs of a row each, as many times over as it has
    /// readers.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        cube_space: &Space,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.plane.estimate(item, cube_space, planes, estimate);
        self.staging(item).count(item, cube_space, estimate);
    }
}

impl AttentionMicrokernel for CubeAttentionMicrokernel {
    /// Its planes' microkernel's.
    fn key_order(&self) -> BindingOrder {
        self.plane.key_order()
    }

    /// Its planes' microkernel's.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        self.plane.served(item)
    }
}
