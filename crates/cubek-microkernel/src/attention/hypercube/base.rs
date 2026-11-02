use crate::attention::cube::{CubeAttentionMicrokernel, CubePlaneReads};
use crate::attention::hypercube::HypercubeAttentionMicrokernelSettings;
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, AttentionWorkItem, G, Q,
};
use crate::split_keys_merge::SlotShape;
use crate::split_keys_merge::hypercube::HypercubeMergeShape;
use core::num::NonZeroUsize;
use cubecl::tile::launch::{Delivery, Vectorization};
use cubecl::tile::{Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{CubePlanes, GlobalStream, KernelEstimate, Microkernel};

/// A box of queries of one head several cubes own together: every cube walks its own run of the
/// keys with the cube microkernel, and each of its planes leaves its queries'
/// partial in a slot of its own, for the routine to merge into the output where it says the
/// partials meet.
///
/// The keys are cut across cubes, and inside each cube across its planes: they are the contracted
/// axis of `p · value` and the columns of `query · keyᵀ`, so each plane scores only its part of its
/// cube's run, and the rows' running max and sum are what let the planes' sums add. No cube waits
/// on another: a cube spinning on a counter another cube moves holds the device until that cube
/// runs, which, where cubes are not preempted, as on Metal, can be never.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAttentionMicrokernel {
    /// What each of its cubes runs on its run of the keys; the plane that merges a box's slots
    /// shares a row's work across its cube's units.
    pub cube: CubeAttentionMicrokernel,
}

impl Microkernel for HypercubeAttentionMicrokernel {
    type WorkItem = AttentionWorkItem;
    type Settings = HypercubeAttentionMicrokernelSettings;
    type Operand = AttentionOperand;
    type CubeSpace = Space;

    const DECLINE: &'static str = "the hypercube attention needs a cube microkernel the device \
         runs and a plane width the device commits to; cubes that stage hold one plane along the \
         keys";

    /// The hypercube's settings over plane fragments, then over plane shuffles
    /// ([`HypercubeAttentionMicrokernelSettings::listed`]).
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [
            HypercubeAttentionMicrokernelSettings::FRAGMENT,
            HypercubeAttentionMicrokernelSettings::SHUFFLE,
        ]
        .into_iter()
        .flat_map(HypercubeAttentionMicrokernelSettings::listed)
        .collect()
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
        let one_along_keys = settings.cube.planes_along_keys.get() == 1;
        CubeAttentionMicrokernel::candidates(hw, item, &settings.cube)
            .filter(move |cube| cube.plane.reads() == CubePlaneReads::Global || one_along_keys)
            .map(|cube| Self { cube })
    }

    /// Its cube's: the slots and the output are written whole lines at a time.
    fn vectorize(&self, cube_space: &Space, vectorization: &mut Vectorization<AttentionOperand>) {
        self.cube.vectorize(cube_space, vectorization);
    }

    /// Its cube's, walking a cube's run of the keys as it would walk them all; then its own: the
    /// cube's partial of each row of its box, a row of the value dim, its max and its sum,
    /// written to its slot and read back by the merge.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        cube_space: &Space,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.cube.estimate(item, cube_space, planes, estimate);
        let rows = (cube_space.extent(Q) * cube_space.extent(G)) as u64;
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

impl HypercubeAttentionMicrokernel {
    /// Bytes of shared memory one cube holds on `hw`, its stages held as `storage`: what its cube
    /// walks with, then what its planes leave their slots through, in the bytes the walk's stages
    /// free. Each cube walks its run once.
    pub fn shared_memory(
        &self,
        hw: &Hardware,
        item: &AttentionWorkItem,
        storage: &StageStorage,
    ) -> usize {
        let partial = self.cube.plane.partial_bytes(item, self.cube.layout());
        self.cube
            .walk_then_shared_memory(hw, item, storage, partial)
    }

    /// What a merge is compiled for, over `item`'s value dim and the `cubes_along_keys` sharing a
    /// box: a slot for every cube, its planes along the keys merged into it before it is
    /// stored, the merging plane's units, and the `seq_q` queries one merging plane holds.
    pub fn merge_shape(
        &self,
        item: &AttentionWorkItem,
        cubes_along_keys: NonZeroUsize,
        seq_q: usize,
    ) -> HypercubeMergeShape {
        let cubes = cubes_along_keys.get();
        HypercubeMergeShape {
            slots: SlotShape {
                units: self.cube.units,
                val_dim: item.val_dim,
                splits: cubes,
            },
            cubes,
            seq_q,
        }
    }
}

/// Its plane, the planes along the keys of a cube where there are several, and the steps a page
/// holds where the cube walks pages.
impl core::fmt::Display for HypercubeAttentionMicrokernel {
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

impl AttentionMicrokernel for HypercubeAttentionMicrokernel {
    /// Its cube's.
    fn key_order(&self) -> BindingOrder {
        self.cube.key_order()
    }

    /// Its cube's.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        self.cube.served(item)
    }

    /// Its cube's.
    fn delivery(&self, widths: &AttentionOperands<usize>, item: &AttentionWorkItem) -> Delivery {
        self.cube.delivery(widths, item)
    }
}
