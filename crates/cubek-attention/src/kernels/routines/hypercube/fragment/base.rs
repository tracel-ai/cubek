use super::kernel::hypercube_fragment_attention_kernel;
use crate::kernels::routines::hypercube::merge::HypercubeSecondLaunch;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionRoutine, AttentionRoutineProblem, AttentionTensors, HypercubeMerge,
};
use crate::kernels::{AttentionPartitioning, QueryOrder};
use core::num::NonZeroUsize;
use cubecl::tile::StageStorage;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with several cubes owning every box of queries, every cube walking its own run of the
/// keys: each cube stages its run's keys and values once for its plane fragments, on a square
/// matrix instruction, each plane leaves its rows undivided in a slot of its own, its softmax
/// state beside them, and a second launch merges the slots into the output, a plane a row. The
/// cubes cover the queries, the heads and the runs, one batch and key head each.
///
/// What a long prompt's few boxes against many keys want: a cube a box would leave the device
/// idle. Its cubes merge by a second launch alone, since a merge on the last arrival writes the
/// output through a binding its planes read as a tile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeSecondLaunchFragmentAttentionRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeSecondLaunchFragmentAttentionRoutineSettings {
    pub microkernel: HypercubeAttentionMicrokernelSettings,
    /// Cubes sharing one box, each on its own run of the keys: part of the partitioning, which
    /// distributes it whatever the keys ([`AttentionPartitioning::new`]).
    pub cubes_along_keys: NonZeroUsize,
}

impl HypercubeSecondLaunchFragmentAttentionRoutineSettings {
    /// Its microkernel's default on eight cubes a box.
    pub const DEFAULT: Self = Self {
        microkernel: HypercubeAttentionMicrokernelSettings::FRAGMENT,
        cubes_along_keys: NonZeroUsize::new(8).unwrap(),
    };
}

impl core::fmt::Display for HypercubeSecondLaunchFragmentAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-second-launch-fragment-attention {} cubes_along_keys={}",
            self.microkernel, self.cubes_along_keys
        )
    }
}

pub type HypercubeSecondLaunchFragmentAttentionRoutineBlueprint =
    AttentionBlueprint<HypercubeAttentionMicrokernel, StageStorage>;

/// The width its planes store their slots' lines at: one element, as the windows their sums
/// drain through serve them.
const SLOT_WIDTH: usize = 1;

impl Routine for HypercubeSecondLaunchFragmentAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = HypercubeAttentionMicrokernel;
    type Settings = HypercubeSecondLaunchFragmentAttentionRoutineSettings;
    type Blueprint = HypercubeSecondLaunchFragmentAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings over plane fragments
    /// ([`HypercubeAttentionMicrokernelSettings::listed`]) on eight cubes a box, then four and
    /// sixteen.
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        [8, 4, 16]
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
        blueprint: &HypercubeSecondLaunchFragmentAttentionRoutineBlueprint,
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
    /// and the shared memory its microkernel asks for, its stages held as its planes read them;
    /// merged by a second launch, that launch checked against the device too. Declines where the
    /// buffers hold too few keys for every cube to walk some
    /// ([`AttentionPartitioning::every_cube_walks`]) or one cube a box already fills the device
    /// ([`AttentionPartitioning::boxes_leave_workers_idle`]).
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if let AttentionCacheLayout::Packed(_) = problem.cache {
            return Err(ATTENTION_PACKED_UNREAD);
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
        partitioning.every_cube_walks()?;
        let storage = microkernel.cube.stage_storage(&partitioning.stage());
        let cube = CubeResources {
            plane_width: hw.plane_dim()?,
            shared_memory: microkernel.shared_memory(hw, &item, &storage),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        partitioning.boxes_leave_workers_idle(hw, &grid.launch)?;
        let shape = microkernel.merge_shape(
            &item,
            partitioning.cubes_along_keys(),
            HypercubeMerge::SecondLaunch.seq_q(&microkernel),
        );
        HypercubeSecondLaunch::new(problem, shape).validate(hw)?;
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

impl AttentionRoutine<HypercubeAttentionMicrokernel, StageStorage>
    for HypercubeSecondLaunchFragmentAttentionRoutine
{
    /// Two launches: every plane leaves its partial in its slot, then a plane for every row
    /// merges the row's slots into the output.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeSecondLaunchFragmentAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        let microkernel = blueprint.microkernel;
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
        let shape = microkernel.merge_shape(
            &problem.work_item(),
            blueprint.partitioning.cubes_along_keys(),
            HypercubeMerge::SecondLaunch.seq_q(&microkernel),
        );
        let slots = tensors.split_slots(shape.slots.splits);
        // Its planes keep their sums in the first and read the stage in the last; they leave
        // their rows in the slots, and the output is read for its shape alone.
        let [sum, _, staged] = microkernel
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
                binding.query.arg(),
                key,
                value,
                binding.out.arg(),
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
                [sum, staged],
            );
        }
        HypercubeSecondLaunch::new(&problem, shape).launch(tensors, slots, SLOT_WIDTH);
    }
}
