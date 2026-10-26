use crate::attention::output::cube::CubeAttentionMicrokernel;
use crate::attention::output::hypercube::{
    HypercubeAttentionMicrokernelSettings, HypercubeAttentionPlaneSettings, HypercubeMerge,
    HypercubeMergeShape,
};
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionPartitioning, AttentionSlotShape,
    AttentionWorkItem, CubePlaneMicrokernel, CubePlaneReads, G, Q,
};
use crate::matmul::MatmulOrder;
use cubecl::tile::StageStorage;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, GlobalStream, KernelEstimate, Microkernel};

/// A box of queries of one head several cubes own together: every cube walks its own run of the
/// keys with the cube microkernel of planes `M`, and each of its planes leaves its queries'
/// partial in a slot of its own, for the routine to merge into the output where it says the
/// partials meet ([`HypercubeMerge`]).
///
/// The keys are cut across cubes, and inside each cube across its planes: they are the contracted
/// axis of `p · value` and the columns of `query · keyᵀ`, so each plane scores only its part of
/// its cube's run, and the rows' running max and sum are what let the planes' sums add. No cube waits
/// on another: a cube spinning on a counter another cube moves holds the device until that cube
/// runs, which, where cubes are not preempted, as on Metal, can be never.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAttentionMicrokernel<M: CubePlaneMicrokernel> {
    /// What each of its cubes runs on its run of the keys; the plane that merges a box's slots
    /// shares a row's work across its cube's units.
    pub cube: CubeAttentionMicrokernel<M>,
}

impl<M: CubePlaneMicrokernel> Microkernel for HypercubeAttentionMicrokernel<M>
where
    M::Settings: HypercubeAttentionPlaneSettings,
{
    type WorkItem = AttentionWorkItem;
    type Settings = HypercubeAttentionMicrokernelSettings<M::Settings>;
    type Operand = AttentionOperand;
    type Partitioning = AttentionPartitioning;

    const DECLINE: &'static str = "the hypercube attention needs a cube microkernel the device \
         runs and a plane width the device commits to; cubes that stage hold one plane along the \
         keys";

    /// The hypercube's settings over planes of its kind
    /// ([`HypercubeAttentionPlaneSettings::hypercube_settings`]).
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        M::Settings::hypercube_settings(hw)
    }

    /// Its cube microkernel's candidates at its cube's settings
    /// ([`CubeAttentionMicrokernel::candidates`]).
    fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        // A staging cube's planes leave their slots through one window of the cube's queries, so
        // planes holding the same queries cannot split their keys.
        let splits =
            M::READS == CubePlaneReads::Global || settings.cube.planes_along_keys.get() == 1;
        CubeAttentionMicrokernel::<M>::candidates(hw, item, &settings.cube)
            .filter(move |_| splits)
            .map(|cube| Self { cube })
    }

    /// Its cube's: the slots and the output are written whole lines at a time.
    fn vectorize(
        &self,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        self.cube.vectorize(partitioning, vectorization);
    }

    /// Its cube's, walking a cube's run of the keys as it would walk them all; then its own: the
    /// cube's partial of each row of its box, a row of the value dim, its max and its sum,
    /// written to its slot and read back by the merge.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        partitioning: &AttentionPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.cube.estimate(item, partitioning, planes, estimate);
        let cube = partitioning.cube_box();
        let rows = (cube.extent(Q) * cube.extent(G)) as u64;
        let size = item.sum.size();
        let partial = GlobalStream {
            bytes: rows * (item.val_dim as u64 + 2) * size as u64,
            run_bytes: item.val_dim * size,
        };
        if let Some(staging) = estimate.staging.as_mut() {
            staging.reads.push(partial);
            staging.writes.push(partial);
        }
    }
}

impl<M: CubePlaneMicrokernel> HypercubeAttentionMicrokernel<M> {
    /// Bytes of shared memory one cube holds on `hw`, its stages held as `storage`: what its cube
    /// walks with, and what its planes leave their slots through.
    pub fn shared_memory(
        &self,
        hw: &Hardware,
        item: &AttentionWorkItem,
        storage: &StageStorage,
    ) -> usize {
        self.cube.walk_shared_memory(hw, item, storage)
            + self.cube.plane.partial_bytes(item, self.cube.layout())
    }

    /// What `merge` is compiled for, over `item`'s value dim and the cubes `partitioning` shares a
    /// box between: a slot for every cube, its planes along the keys merged into it before it is
    /// stored, the merging plane's units, and the queries one merging plane holds, a box's for the
    /// last arrival and one for a second launch.
    pub fn merge_shape(
        &self,
        item: &AttentionWorkItem,
        partitioning: &AttentionPartitioning,
        merge: HypercubeMerge,
    ) -> HypercubeMergeShape {
        let cubes = partitioning.cubes_along_keys().get();
        HypercubeMergeShape {
            slots: AttentionSlotShape {
                units: self.cube.units,
                val_dim: item.val_dim,
                splits: cubes,
            },
            cubes,
            seq_q: match merge {
                HypercubeMerge::LastArrival => self.cube.seq_q(),
                HypercubeMerge::SecondLaunch => 1,
            },
        }
    }
}

/// Its plane, the planes along the keys of a cube where there are several, and the steps a page
/// holds where the cube walks pages.
impl<M: CubePlaneMicrokernel> core::fmt::Display for HypercubeAttentionMicrokernel<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.cube.plane)?;
        let planes = self.cube.layout().planes_along_keys;
        if planes > 1 {
            write!(f, " planes_along_keys={planes}")?;
        }
        if let Some(steps) = self.cube.page_steps {
            write!(f, " page_steps={steps}")?;
        }
        Ok(())
    }
}

impl<M: CubePlaneMicrokernel> AttentionMicrokernel for HypercubeAttentionMicrokernel<M>
where
    M::Settings: HypercubeAttentionPlaneSettings,
{
    /// Its cube's.
    fn key_order(&self) -> MatmulOrder {
        self.cube.key_order()
    }

    /// Its cube's.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        self.cube.served(item)
    }
}
