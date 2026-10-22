use super::kernel::cube_shuffle_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionRoutine, AttentionRoutineProblem, AttentionTensors,
};
use cubecl::tile::StageStorage;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, CubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::plane::{
    PlaneShuffleAttentionMicrokernel, PlaneShuffleAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{AttentionOperand, AttentionPartitioning, QueryOrder};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with every cube owning a box of queries of some heads of the group, and walking
/// nothing else: each plane of the cube owns its rows of one head and reads its keys and values
/// where they lie, its units scoring a key in teams and summing their partial dots across the
/// plane, and the cube's planes along the keys merging their rows in shared memory. The cubes
/// cover the queries and the heads, one batch and key head each.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeShuffleAttentionRoutine;

/// The cube microkernel it places: a cube of plane shuffles.
pub type CubeShuffleAttentionMicrokernel =
    CubeAttentionMicrokernel<PlaneShuffleAttentionMicrokernel>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeShuffleAttentionRoutineSettings {
    pub microkernel: CubeAttentionMicrokernelSettings<PlaneShuffleAttentionMicrokernelSettings>,
    /// The order the cubes take the boxes of queries in: part of the partitioning
    /// ([`AttentionPartitioning::new`]).
    pub queries: QueryOrder,
}

impl CubeShuffleAttentionRoutineSettings {
    /// Four planes of the plane shuffle's default splitting one head's keys.
    pub const DEFAULT: Self = Self {
        microkernel:
            CubeAttentionMicrokernelSettings::<PlaneShuffleAttentionMicrokernelSettings>::DEFAULT,
        queries: QueryOrder::InOrder,
    };
}

impl core::fmt::Display for CubeShuffleAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-shuffle-attention {}{}",
            self.microkernel, self.queries
        )
    }
}

pub type CubeShuffleAttentionRoutineBlueprint =
    AttentionBlueprint<CubeShuffleAttentionMicrokernel, StageStorage>;

impl Routine for CubeShuffleAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = CubeShuffleAttentionMicrokernel;
    type Settings = CubeShuffleAttentionRoutineSettings;
    type Blueprint = CubeShuffleAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings ([`CubeAttentionMicrokernel::settings`]) in query order, then
    /// its microkernel's default with the boxes taken last first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeAttentionMicrokernel::<PlaneShuffleAttentionMicrokernel>::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                queries: QueryOrder::InOrder,
            })
            .chain([Self::Settings {
                queries: QueryOrder::LastFirst,
                ..Self::Settings::DEFAULT
            }])
            .collect()
    }
    /// Nothing of its own: it walks cubes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeShuffleAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Nothing of its own: the cube microkernel places the staging and its planes the work.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &CubeShuffleAttentionRoutineBlueprint,
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

    /// The best cube of plane shuffles at the settings, a cube a box, holding a plane for every
    /// head it holds along every plane of its queries and of their keys, each as many units as the
    /// device's plane, and the shared memory the cube microkernel asks for, the cubes taking the
    /// boxes in the settings' order.
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
        let microkernel = CubeShuffleAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
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

impl AttentionRoutine<CubeShuffleAttentionMicrokernel, StageStorage>
    for CubeShuffleAttentionRoutine
{
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: CubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
        // The plane shuffle lines every operand alike, and the kernel binds them at one width.
        assert!(
            [
                blueprint.widths.key,
                blueprint.widths.value,
                blueprint.widths.out
            ]
            .iter()
            .all(|&width| width == blueprint.widths.query),
            "the plane shuffle's operands share one width"
        );
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, the keys and values are read no further than
        // the attended length, and the kernel indexes no buffer but through its tiles and the
        // length's first element; the slots the planes merge in are the cube's own.
        unsafe {
            cube_shuffle_attention_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.widths.query,
                binding.query.arg(),
                key.arg(),
                value.arg(),
                binding.out.arg(),
                binding.lengths,
                binding.pages,
                problem.seq_q as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                problem.val_dim,
                problem.visibility,
                problem.work_item().kernel_elements(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::AttentionCacheLayout;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::attention::S;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A decode step, a few positions of a prompt whose queries divide nothing, and a wide group
    /// over a long cache in `f32`.
    fn problems() -> [AttentionRoutineProblem; 3] {
        [
            AttentionRoutineProblem::dense((1, 32, 8), (1, 2048), (128, 128), F16),
            AttentionRoutineProblem::dense((2, 8, 2), (5, 300), (64, 64), F16),
            AttentionRoutineProblem::dense((1, 16, 1), (1, 4096), (128, 128), F32),
        ]
    }

    /// Every setting prepares where the device commits to a plane width, with no matrix
    /// instruction at all; one other than the default may ask for more than a cube holds, and
    /// says so.
    #[test]
    fn every_setting_prepares_where_planes_run() {
        for hw in [devices::m2_pro(), devices::h100()] {
            for (position, settings) in CubeShuffleAttentionRoutine.settings(&hw).iter().enumerate()
            {
                for problem in problems() {
                    let prepared = CubeShuffleAttentionRoutine.prepare(&hw, &problem, settings);
                    let outgrown = matches!(
                        prepared,
                        Err(Refusal::SharedMemoryPastDevice { .. } | Refusal::CubePastDevice { .. })
                    );
                    assert!(
                        prepared.is_ok() || (position > 0 && outgrown),
                        "{settings} on {hw:?}: {prepared:?}"
                    );
                }
            }
        }
    }

    /// Every setting declines where the device commits to no plane width, rather than launching
    /// on lanes it would misread.
    #[test]
    fn every_setting_declines_without_a_plane_width() {
        for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
            for settings in CubeShuffleAttentionRoutine.settings(&devices::m2_pro()) {
                for problem in problems() {
                    let declined = CubeShuffleAttentionRoutine.prepare(&hw, &problem, &settings);
                    assert!(declined.is_err(), "{settings} on {hw:?}: {declined:?}");
                }
            }
        }
    }

    /// A cube over pages of 64 positions walks a page at a time, every page holding several of
    /// its steps: it still stages one step, and its box is still the whole sequence.
    #[test]
    fn a_cube_over_pages_stages_one_step() {
        let [decode, ..] = problems();
        let paged = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Pages { page_size: 64 },
            ..decode
        };
        let settings = &CubeShuffleAttentionRoutine.settings(&devices::m2_pro())[0];
        let (blueprint, _) = CubeShuffleAttentionRoutine
            .prepare(&devices::m2_pro(), &paged, settings)
            .expect("the lead prepares on the M2");
        let keys = blueprint.microkernel.seq_kv();
        assert_eq!(
            blueprint.microkernel.page_steps.map(|steps| steps.get()),
            Some(64 / keys)
        );
        assert_eq!(blueprint.partitioning.stage().extent(S), keys);
        assert_eq!(blueprint.partitioning.cube_box().extent(S), 2048);
    }
}
