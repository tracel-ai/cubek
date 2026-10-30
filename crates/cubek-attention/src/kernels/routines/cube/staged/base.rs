use super::kernel::cube_staged_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionPartitioning, AttentionRoutine, AttentionRoutineProblem, AttentionTensors, QueryOrder,
};
use cubecl::tile::StageStorage;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, CubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with every cube owning a box of queries of some heads of the group, and walking
/// nothing else: the cube microkernel stages the keys and values once for its planes, each plane
/// owning its queries of one head. The cubes cover the queries and the heads, one batch and key
/// head each; a group wider than a cube's heads reads its keys once a cube.
///
/// Its planes are fragments of a square matrix instruction, which read the stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedAttentionRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedAttentionRoutineSettings {
    pub microkernel: CubeAttentionMicrokernelSettings,
    /// The order the cubes take the boxes of queries in: part of the partitioning
    /// ([`AttentionPartitioning::new`]).
    pub queries: QueryOrder,
}

pub type CubeStagedAttentionRoutineBlueprint =
    AttentionBlueprint<CubeAttentionMicrokernel, StageStorage>;

impl CubeStagedAttentionRoutineSettings {
    /// The cube's lead over plane fragments, its boxes taken in order.
    pub const DEFAULT: Self = Self {
        microkernel: CubeAttentionMicrokernelSettings::FRAGMENT,
        queries: QueryOrder::InOrder,
    };
}

impl core::fmt::Display for CubeStagedAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-{}-attention {}{}",
            self.microkernel.plane.name(),
            self.microkernel,
            self.queries
        )
    }
}

impl Routine for CubeStagedAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = CubeAttentionMicrokernel;
    type Settings = CubeStagedAttentionRoutineSettings;
    type Blueprint = CubeStagedAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings over plane fragments
    /// ([`CubeAttentionMicrokernelSettings::listed`]) in query order, then the first of them, its
    /// lead, with the boxes taken last first.
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        let microkernel = CubeAttentionMicrokernelSettings::FRAGMENT.listed();
        let last_first = Self::Settings {
            microkernel: microkernel[0],
            queries: QueryOrder::LastFirst,
        };
        microkernel
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                queries: QueryOrder::InOrder,
            })
            .chain([last_first])
            .collect()
    }

    /// Nothing of its own: it walks cubes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(&partitioning.cube_space(), vectorization);
    }

    /// Nothing of its own: the cube microkernel places the staging and its planes the work.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &CubeStagedAttentionRoutineBlueprint,
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

    /// The best cube at the settings, a cube a box, holding a plane for every
    /// head it holds along every plane of its queries and of their keys, each as many units as
    /// the device's plane, and the shared memory the cube microkernel asks for, its stages held as
    /// its planes read them, the cubes taking the boxes in the settings' order. Its planes read
    /// the stage as it is laid out, and merge their queries where they split the keys.
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
        let microkernel = CubeAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
        let partitioning = AttentionPartitioning::new(
            problem.space(),
            microkernel.tiling(),
            microkernel.page_steps,
            AttentionPartitioning::ONE_CUBE_A_BOX,
            settings.queries,
        );
        let storage = microkernel.stage_storage(&partitioning.stage());
        let cube = CubeResources {
            plane_width: hw.plane_dim()?,
            shared_memory: microkernel.shared_memory(hw, &item, &storage),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        microkernel.plane.reads_stage(settings.microkernel.layout)?;
        if settings.microkernel.planes_along_keys.get() > 1 {
            microkernel.plane.merges_rows()?;
        }
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

impl AttentionRoutine<CubeAttentionMicrokernel, StageStorage> for CubeStagedAttentionRoutine {
    /// Every cube staging its keys and values once for its planes, which read the stage, held as
    /// the blueprint's storage.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: CubeStagedAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
        // Its planes keep their sums in the first and read the stage in the last; the output is
        // stored as its tile states.
        let [sum, _, staged] = blueprint
            .microkernel
            .served(&problem.work_item())
            .served_kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, the keys and values are read no further than
        // the attended length, and the kernel indexes no buffer but through its tiles and the
        // length's first element; each plane's windows are its own.
        unsafe {
            cube_staged_attention_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                binding.query.arg(),
                key,
                value,
                binding.out.arg(),
                binding.logsumexp.map(|logsumexp| logsumexp.arg()).into(),
                binding.mask.map(|mask| mask.arg()).into(),
                binding.lengths,
                problem.seq_q as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.shared_memory.clone(),
                problem.visibility,
                [sum, staged],
            );
        }
    }
}
