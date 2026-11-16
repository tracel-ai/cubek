use super::kernel::hypercube_fragment_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionPartitioning, AttentionRoutineProblem, AttentionSplitSlots, AttentionTensors,
    HypercubeMerge, PACKED_QUERIES_UNREAD, PartialAttentionRoutine, QueryOrder,
};
use core::num::NonZeroUsize;
use cubecl::tile::StageStorage;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionOperand};
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Partial attention, with several cubes owning every box of queries, every cube walking its own
/// run of the keys: each cube stages its run's keys and values once for its plane fragments, on a
/// square matrix instruction, and each plane leaves its queries undivided in a slot of its own, its
/// softmax state beside them. It writes no output: the split-keys merge launched after it merges
/// the slots into the output, a plane a query of a head. The cubes cover the queries, the heads and the runs,
/// one batch and key head each.
///
/// What a long prompt's few boxes against many keys want: a cube a box would leave the device
/// idle. Its cubes have no merge of their own on the last arrival, since that merge writes the
/// output through a binding its planes read as a tile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeFragmentPartialAttentionRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeFragmentPartialAttentionRoutineSettings {
    pub microkernel: HypercubeAttentionMicrokernelSettings,
    /// Cubes sharing one box, each on its own run of the keys: part of the partitioning, which
    /// distributes it whatever the keys ([`AttentionPartitioning::new`]).
    pub cubes_along_keys: NonZeroUsize,
}

impl HypercubeFragmentPartialAttentionRoutineSettings {
    /// Its microkernel's default on eight cubes a box.
    pub const DEFAULT: Self = Self {
        microkernel: HypercubeAttentionMicrokernelSettings::FRAGMENT,
        cubes_along_keys: NonZeroUsize::new(8).unwrap(),
    };
}

impl core::fmt::Display for HypercubeFragmentPartialAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-fragment-partial-attention {} cubes_along_keys={}",
            self.microkernel, self.cubes_along_keys
        )
    }
}

pub type HypercubeFragmentPartialAttentionRoutineBlueprint =
    AttentionBlueprint<HypercubeAttentionMicrokernel, StageStorage>;

/// The width its planes store their slots' lines at: one element, as the windows their sums
/// drain through serve them.
const SLOT_WIDTH: usize = 1;

impl Routine for HypercubeFragmentPartialAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = HypercubeAttentionMicrokernel;
    type Settings = HypercubeFragmentPartialAttentionRoutineSettings;
    type Blueprint = HypercubeFragmentPartialAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings over plane fragments
    /// ([`HypercubeAttentionMicrokernelSettings::listed`]) on eight cubes a box, then four and
    /// sixteen, then seven, ten and thirteen: counts that are no power of two, which may fill the
    /// device's last wave of boxes where a power of two leaves it mostly idle.
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        [8, 4, 16, 7, 10, 13]
            .into_iter()
            .flat_map(|cubes| {
                HypercubeAttentionMicrokernelSettings::FRAGMENT
                    .listed()
                    .into_iter()
                    .map(move |microkernel| Self::Settings {
                        microkernel,
                        cubes_along_keys: NonZeroUsize::new(cubes).unwrap(),
                    })
            })
            .collect()
    }

    /// Nothing of its own: it only distributes the cubes, and their microkernel makes every
    /// access.
    fn vectorize(
        &self,
        microkernel: &HypercubeAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(&partitioning.cube_space(), vectorization);
    }

    /// Nothing of its own: its cubes' microkernel walks each cube's run of the keys and counts
    /// each cube's partial.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeFragmentPartialAttentionRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint.microkernel.estimate(
            &problem.work_item(),
            &blueprint.partitioning.cube_space(),
            grid.planes(),
            estimate,
        );
    }

    /// The best hypercube of plane fragments at the settings, its settings' cubes on every box
    /// whatever the keys ([`AttentionPartitioning::new`]), each holding its planes along the keys
    /// and the shared memory its microkernel asks for, its stages held as its planes read them.
    /// Serves a cache of any length, at any device fill: neither is a call's to decline, since an
    /// autotune key prepares its candidates at one representative for every call of its bucket.
    /// The merge after it is prepared by the strategy that launches it.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if let AttentionCacheLayout::Packed(_) = problem.cache {
            return Err(ATTENTION_PACKED_UNREAD);
        }
        if problem.packed_queries {
            return Err(PACKED_QUERIES_UNREAD);
        }
        let item = problem.work_item();
        let microkernel = HypercubeAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
        microkernel
            .cube
            .plane
            .reads_stage(microkernel.cube.settings.layout)?;
        let partitioning = AttentionPartitioning::new(
            problem.space(),
            microkernel.cube.tiling(),
            microkernel.cube.page_steps,
            settings.cubes_along_keys,
            QueryOrder::InOrder,
        );
        let storage = microkernel.cube.stage_storage(&partitioning.stage());
        let cube = CubeResources {
            plane_width: hw.plane_dim()?,
            shared_memory: microkernel.shared_memory(hw, &item, &storage),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint = AttentionBlueprint::new(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            storage,
        );
        Ok((blueprint, grid))
    }
}

impl PartialAttentionRoutine<HypercubeAttentionMicrokernel, StageStorage>
    for HypercubeFragmentPartialAttentionRoutine
{
    fn merge_shape(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeFragmentPartialAttentionRoutineBlueprint,
    ) -> HypercubeMergeShape {
        blueprint.microkernel.merge_shape(
            &problem.work_item(),
            blueprint.partitioning.cubes_along_keys(),
            HypercubeMerge::SecondLaunch.seq_q(&blueprint.microkernel),
        )
    }

    fn slot_width(&self, _blueprint: &HypercubeFragmentPartialAttentionRoutineBlueprint) -> usize {
        SLOT_WIDTH
    }

    /// Every plane leaving its partial in its slot.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeFragmentPartialAttentionRoutineBlueprint,
        grid: Grid,
        slots: &AttentionSplitSlots,
    ) {
        let problem = tensors.problem();
        let shape = self.merge_shape(&problem, &blueprint);
        let microkernel = blueprint.microkernel;
        let (key, value) = binding.cache.into_tiles();
        // Its planes keep their sums in the first, read the query and the keys' stage in the
        // second and cast their probabilities to the values' element in the last; they leave
        // their rows in the slots, and the output is read for its shape alone.
        let [sum, _, staged, probabilities] = microkernel
            .served(&problem.work_item())
            .served_kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads of a tile the
        // problem's edge cuts short, the keys and values are read no further than the attended
        // length, and the output is read for its shape alone. A plane writes no slot but its own,
        // for the rows of its box the problem holds.
        unsafe {
            hypercube_fragment_attention_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                binding.query,
                key,
                value,
                binding.out,
                slots.out.clone().binding().into_tensor_arg(),
                slots.max.clone().binding().into_tensor_arg(),
                slots.sum.clone().binding().into_tensor_arg(),
                binding.mask.map(|mask| mask.arg()).into(),
                binding.lengths,
                problem.seq_q as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                microkernel,
                shape,
                problem.visibility,
                blueprint.shared_memory.clone(),
                [sum, staged, probabilities],
            );
        }
    }
}
