use super::kernel::cube_staged_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionRoutine, AttentionRoutineProblem, AttentionTensors,
};
use core::marker::PhantomData;
use cubecl::tile::StageStorage;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::attention::output::cube::{
    CubeAttentionMicrokernel, CubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::output::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneUnitGroupAttentionMicrokernel,
};
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionPartitioning, CubePlaneMicrokernel,
    CubeStagedPlane, QueryOrder,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with every cube owning a box of queries of some heads of the group, and walking
/// nothing else: the cube microkernel stages the keys and values once for its planes of `P`, each
/// plane owning its queries of one head. The cubes cover the queries and the heads, one batch and
/// key head each; a group wider than a cube's heads reads its keys once a cube.
///
/// Its planes are fragments of a square matrix instruction
/// ([`PlaneFragmentAttentionMicrokernel`]), or groups of units each owning its queries in
/// registers on no matrix instruction ([`PlaneUnitGroupAttentionMicrokernel`]), what a cube runs
/// where the device has no square instruction for the element, as `bf16` on Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeStagedAttentionRoutine<P> {
    plane: PhantomData<P>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedAttentionRoutineSettings<P: CubePlaneMicrokernel> {
    pub microkernel: CubeAttentionMicrokernelSettings<P::Settings>,
    /// The order the cubes take the boxes of queries in: part of the partitioning
    /// ([`AttentionPartitioning::new`]).
    pub queries: QueryOrder,
}

pub type CubeStagedAttentionRoutineBlueprint<P> =
    AttentionBlueprint<CubeAttentionMicrokernel<P>, StageStorage>;

impl<P> CubeStagedAttentionRoutine<P> {
    pub const fn new() -> Self {
        Self { plane: PhantomData }
    }
}

impl<P> Default for CubeStagedAttentionRoutine<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl CubeStagedAttentionRoutineSettings<PlaneFragmentAttentionMicrokernel> {
    pub const DEFAULT: Self = Self {
        microkernel: CubeAttentionMicrokernelSettings::<
            <PlaneFragmentAttentionMicrokernel as Microkernel>::Settings,
        >::DEFAULT,
        queries: QueryOrder::InOrder,
    };
}

impl CubeStagedAttentionRoutineSettings<PlaneUnitGroupAttentionMicrokernel> {
    pub const DEFAULT: Self = Self {
        microkernel: CubeAttentionMicrokernelSettings::<
            <PlaneUnitGroupAttentionMicrokernel as Microkernel>::Settings,
        >::DEFAULT,
        queries: QueryOrder::InOrder,
    };
}

impl<P: CubePlaneMicrokernel> core::fmt::Display for CubeStagedAttentionRoutineSettings<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-{}-attention {}{}",
            P::NAME,
            self.microkernel,
            self.queries
        )
    }
}

impl<P: CubePlaneMicrokernel + CubeStagedPlane> Routine for CubeStagedAttentionRoutine<P> {
    type Problem = AttentionRoutineProblem;
    type Microkernel = CubeAttentionMicrokernel<P>;
    type Settings = CubeStagedAttentionRoutineSettings<P>;
    type Blueprint = CubeStagedAttentionRoutineBlueprint<P>;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings ([`CubeAttentionMicrokernel::settings`]) in query order, then
    /// the first of them, its microkernel's lead, with the boxes taken last first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let microkernel = CubeAttentionMicrokernel::<P>::settings(hw);
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
        microkernel: &CubeAttentionMicrokernel<P>,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Nothing of its own: the cube microkernel places the staging and its planes the work.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &CubeStagedAttentionRoutineBlueprint<P>,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint.microkernel.estimate(
            &problem.work_item(),
            &blueprint.partitioning,
            grid.planes(),
            estimate,
        );
    }

    /// The best cube of `P`'s planes at the settings, a cube a box, holding a plane for every
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
        let microkernel = CubeAttentionMicrokernel::<P>::best(hw, &item, &settings.microkernel)?;
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

impl<P: CubePlaneMicrokernel + CubeStagedPlane>
    AttentionRoutine<CubeAttentionMicrokernel<P>, StageStorage> for CubeStagedAttentionRoutine<P>
{
    /// Every cube staging its keys and values once for its planes, which read the stage, held as
    /// the blueprint's storage.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: CubeStagedAttentionRoutineBlueprint<P>,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
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
                binding.lengths,
                problem.seq_q as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.shared_memory.clone(),
                problem.visibility,
                blueprint
                    .microkernel
                    .served(&problem.work_item())
                    .served_kernel_elements(),
            );
        }
    }
}
