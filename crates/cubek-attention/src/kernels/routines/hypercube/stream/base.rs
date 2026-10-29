use super::kernel::hypercube_stream_fragment_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionPartitioning, AttentionRoutine, AttentionRoutineProblem, AttentionTensors,
    HypercubeMerge,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Space, StageStorage};
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionOperand, S};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with every batch, key head and box of queries' steps of keys laid end to end on one
/// line, shared between a grid sized to the device's workers, one run each: a run may start inside
/// one box and end inside another, and every run but the last holds the same steps, so no cube
/// walks more than its share however the boxes divide the grid
/// ([`AttentionPartitioning::shared`]). Each cube stages its run's keys and values for plane
/// fragments on a square matrix instruction, as the cube routine does.
///
/// What a long prompt's boxes against many keys want, where they are more than the device holds
/// at once but not a whole number of times over: one cube a box, or a fixed count a box, leaves
/// the device idle through the last round of boxes, where a grid sized to the device runs no such
/// round. A box its run holds whole is written into the output as the cube routine writes one; a
/// part of a box several runs share is left in a slot, and the box's last run to arrive merges the
/// slots into the output ([`HypercubeMerge::LastArrival`]). No cube waits on another. Declines
/// where one cube's stores do not reach another within a launch, and on a device that does not say
/// how many workers it has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeStreamFragmentAttentionRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeStreamFragmentAttentionRoutineSettings {
    pub microkernel: HypercubeAttentionMicrokernelSettings,
    /// Cubes the grid holds for every worker of the device.
    pub cubes_per_worker: NonZeroUsize,
}

impl HypercubeStreamFragmentAttentionRoutineSettings {
    /// Its microkernel's default, one cube a worker: a staging cube of plane fragments takes most
    /// of a worker's shared memory, so one is what a worker runs at once, and the grid is one round.
    pub const DEFAULT: Self = Self {
        microkernel: HypercubeAttentionMicrokernelSettings::FRAGMENT,
        cubes_per_worker: NonZeroUsize::MIN,
    };
}

impl core::fmt::Display for HypercubeStreamFragmentAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-stream-fragment-attention {} cubes_per_worker={}",
            self.microkernel, self.cubes_per_worker
        )
    }
}

pub type HypercubeStreamFragmentAttentionRoutineBlueprint =
    AttentionBlueprint<HypercubeAttentionMicrokernel, StageStorage>;

impl HypercubeStreamFragmentAttentionRoutine {
    /// What it declines on a device that does not say how many workers it has, the count its grid
    /// is sized by.
    pub const UNSIZED: Refusal = Refusal::Declined {
        why: "the stream routine sizes its grid by the device's workers, which this device does \
              not report",
    };

    /// The cubes its grid holds on `hw` at `settings`, or [`UNSIZED`](Self::UNSIZED) on a device
    /// that does not say how many workers it has.
    fn cubes(
        hw: &Hardware,
        settings: &HypercubeStreamFragmentAttentionRoutineSettings,
    ) -> Result<NonZeroUsize, Refusal> {
        let workers = hw
            .workers()
            .and_then(|workers| NonZeroUsize::new(workers as usize))
            .ok_or(Self::UNSIZED)?;
        Ok(workers.saturating_mul(settings.cubes_per_worker))
    }

    /// Slots every row holds: as many as runs can share a box. A run holds at least
    /// `boxes / cubes` boxes' steps, so at most `cubes / boxes` runs fit inside one box, and one
    /// more ends in it.
    fn splits(problem: &AttentionRoutineProblem, partitioning: &AttentionPartitioning) -> usize {
        let boxes = problem.batch * problem.num_kv_heads * partitioning.boxes();
        partitioning.cubes().div_ceil(boxes) + 1
    }
}

impl Routine for HypercubeStreamFragmentAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = HypercubeAttentionMicrokernel;
    type Settings = HypercubeStreamFragmentAttentionRoutineSettings;
    type Blueprint = HypercubeStreamFragmentAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings over plane fragments
    /// ([`HypercubeAttentionMicrokernelSettings::listed`]) on one cube a worker, then two.
    /// Nothing where one cube's stores do not reach another within a launch, or where the device
    /// does not say how many workers it has, where every one of them declines.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if HypercubeMerge::LastArrival.admits(hw).is_err() || hw.workers().is_none() {
            return Vec::new();
        }
        [1, 2]
            .into_iter()
            .flat_map(|cubes| {
                HypercubeAttentionMicrokernelSettings::FRAGMENT
                    .listed()
                    .into_iter()
                    .map(move |microkernel| Self::Settings {
                        microkernel,
                        cubes_per_worker: NonZeroUsize::new(cubes).unwrap(),
                    })
            })
            .collect()
    }

    /// Its cubes' microkernel's, and the output an element at a time: the last arrival merges a
    /// row's slots one element a unit, as the slots are written, and writes them as it holds them.
    fn vectorize(
        &self,
        microkernel: &HypercubeAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        vectorization.require(VectorRequirement::Elementwise {
            operand: AttentionOperand::Out,
        });
        microkernel.vectorize(&partitioning.cube_space(), vectorization);
    }

    /// Its cubes' microkernel's, over one cube's run as if it were one box's keys: a run reads as
    /// many steps of keys and values as that, whichever boxes they lie in, and leaves a partial
    /// for each box it shares, which is at most two.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeStreamFragmentAttentionRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let partitioning = &blueprint.partitioning;
        let cube = partitioning.cube_space();
        let step = partitioning.stage().extent(S);
        let boxes = problem.batch * problem.num_kv_heads * partitioning.boxes();
        let line = boxes * problem.seq_kv.div_ceil(step);
        let run = line.div_ceil(partitioning.cubes()) * step;
        let run_space = Space::new(
            &cube
                .extents()
                .into_iter()
                .map(|(axis, extent)| (axis, if axis == S { run } else { extent }))
                .collect::<Vec<_>>(),
        );
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &run_space, grid.planes(), estimate);
    }

    /// The best hypercube of plane fragments at the settings, its grid the device's workers times
    /// the setting's count, sharing the line of every box's steps
    /// ([`AttentionPartitioning::shared`]), each cube holding the shared memory its microkernel
    /// asks for, its stages held as its planes read them, and the word it learns it came last in.
    /// Declines a cache of pages, which a run would walk across, and a packed one.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if let AttentionCacheLayout::Packed(_) = problem.cache {
            return Err(ATTENTION_PACKED_UNREAD);
        }
        HypercubeMerge::LastArrival.admits(hw)?;
        let item = problem.work_item();
        let microkernel = HypercubeAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
        microkernel
            .cube
            .plane
            .reads_stage(microkernel.cube.settings.layout)?;
        if microkernel.cube.page_steps.is_some() {
            return Err(Refusal::Declined {
                why: "the stream routine's runs cross from one box's keys into the next's, which \
                      a cache of pages does not lie in",
            });
        }
        let partitioning = AttentionPartitioning::shared(
            problem.space(),
            microkernel.cube.tiling(),
            Self::cubes(hw, settings)?,
        );
        let storage = microkernel.cube.stage_storage(&partitioning.stage());
        let cube = CubeResources {
            plane_width: hw.plane_dim()?,
            shared_memory: microkernel.shared_memory(hw, &item, &storage)
                + HypercubeMerge::LastArrival.shared_memory(),
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

impl AttentionRoutine<HypercubeAttentionMicrokernel, StageStorage>
    for HypercubeStreamFragmentAttentionRoutine
{
    /// One launch: every cube walks its run, writing a box it holds whole into the output and
    /// leaving its part of a shared one in its slot, the box's last arrival merging the slots.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeStreamFragmentAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        let microkernel = blueprint.microkernel;
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
        let splits = Self::splits(&problem, &blueprint.partitioning);
        let slots = tensors.split_slots(splits);
        let arrivals = tensors.box_arrivals(microkernel.cube.seq_q());
        // Its planes keep their sums in the first, write the output in the second and read the
        // stage in the last.
        let elements = microkernel
            .served(&problem.work_item())
            .served_kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the keys and values are read no further
        // than the attended length. A cube writes the output of a box its run holds whole, which
        // no other cube touches, and no slot but its own of a box it shares; the last cube of a
        // box reads the box's slots, acquired after every holder stored its own, and writes its
        // output. A box's counter, one of the line's boxes, which the arrivals outnumber, holds
        // zero, and every launch leaves it so.
        unsafe {
            hypercube_stream_fragment_attention_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                binding.query.arg(),
                key,
                value,
                binding.out.arg(),
                binding.logsumexp.map(|logsumexp| logsumexp.arg()).into(),
                slots.out.binding().into_tensor_arg(),
                slots.max.binding().into_tensor_arg(),
                slots.sum.binding().into_tensor_arg(),
                arrivals.binding.into_tensor_arg().into_buffer_arg(),
                binding.mask.map(|mask| mask.arg()).into(),
                binding.lengths,
                problem.seq_q as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                microkernel,
                splits,
                problem.visibility,
                blueprint.shared_memory.clone(),
                elements,
            );
        }
    }
}
