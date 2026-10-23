use super::kernel::unit_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionRoutine, AttentionRoutineProblem, AttentionTensors,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::UnitCube;
use cubek_microkernel::attention::unit::{
    UnitAttentionMicrokernel, UnitAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{
    AttentionOperand, AttentionOperands, AttentionPartitioning, AttentionStagingCount, Q, S,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with every unit owning rows of one head. This kind's floor: it asks nothing of the
/// hardware, so it runs everywhere.
///
/// The launch's units cover the queries and the heads, each its microkernel's rows of one head,
/// one batch and key head each, taken in turn along the queries of a head and on into the next
/// head, so a decode step's single row of each head still fills a cube. Which cube or plane a unit
/// sits in is only how the launch groups them. Every unit reads its query, keys and values where
/// they lie and holds nothing in shared memory: no unit waits on another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitAttentionRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitAttentionRoutineSettings {
    pub microkernel: UnitAttentionMicrokernelSettings,
    /// Planes the launch groups its units in, a cube.
    pub planes: NonZeroUsize,
}

impl UnitAttentionRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: UnitAttentionMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(2).unwrap(),
    };
}

impl core::fmt::Display for UnitAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-attention {} planes={}",
            self.microkernel, self.planes
        )
    }
}

pub type UnitAttentionRoutineBlueprint = AttentionBlueprint<UnitAttentionMicrokernel>;

impl Routine for UnitAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = UnitAttentionMicrokernel;
    type Settings = UnitAttentionRoutineSettings;
    type Blueprint = UnitAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// Its microkernel's settings ([`UnitAttentionMicrokernel::settings`]), each at the routine's
    /// default, then a cube as full as the device holds ([`Hardware::full_cube_planes`]).
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = UnitAttentionRoutineSettings::DEFAULT;
        let full = Self::Settings {
            planes: NonZeroUsize::new(hw.full_cube_planes()).unwrap(),
            ..lead
        };
        UnitAttentionMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                ..lead
            })
            .chain([full])
            .collect()
    }

    /// Nothing of its own: it walks the launch's units, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &UnitAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Its units' steps: each unit reads its rows of the query once a step, and every key and
    /// value of the walk, straight from global memory — each unit of the cube reading them again —
    /// and writes its rows once. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &UnitAttentionRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, partitioning) = (problem.work_item(), &blueprint.partitioning);
        let (cube, leaf) = (partitioning.cube_box(), partitioning.leaf());
        let steps = cube.extent(S).div_ceil(leaf.extent(S)) as u64;
        let row_readers = (cube.extent(Q) / leaf.extent(Q)) as u64;
        AttentionStagingCount {
            seq_kv: leaf.extent(S),
            stages_in_flight: 1,
            runs: AttentionOperands {
                query: item.head_dim,
                key: leaf.extent(S),
                value: item.val_dim,
                out: item.val_dim,
            },
            readers: AttentionOperands {
                query: steps,
                key: row_readers,
                value: row_readers,
                out: 1,
            },
        }
        .count(&item, partitioning, estimate);
        blueprint
            .microkernel
            .estimate(&item, partitioning, grid.planes(), estimate);
    }

    /// The units cooperate on nothing, so a cube groups its settings' planes, each as wide as the
    /// device dispatches one and no wider than the cube's unit budget leaves for each: any width
    /// is correct. The width sizes the grid and the cubes the units are grouped in.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if let AttentionCacheLayout::Packed(_) = problem.cache {
            return Err(ATTENTION_PACKED_UNREAD);
        }
        let microkernel =
            UnitAttentionMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let planes = settings.planes.get();
        let plane_width = (hw.properties.hardware.max_units_per_cube / planes as u32)
            .min(hw.dispatch_plane_size())
            .max(1);
        let partitioning = AttentionPartitioning::over_units(
            problem.space(),
            microkernel.tiling(),
            UnitCube {
                plane_width: plane_width as usize,
                planes,
            },
        );
        let cube = CubeResources {
            plane_width,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint =
            AttentionBlueprint::new(hw, problem, microkernel, partitioning, vectorization, ());
        Ok((blueprint, grid))
    }
}

impl AttentionRoutine<UnitAttentionMicrokernel> for UnitAttentionRoutine {
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: UnitAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, the keys and values are read no further than
        // the attended length, and the kernel indexes no buffer but through its tiles and the
        // length's first element.
        unsafe {
            unit_attention_kernel::launch_unchecked(
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
                problem.visibility,
                problem.work_item().sum,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `hw` offering the square instruction of `edge` for `f16` and for `f32`, summed in `f32`.
    fn offering(hw: Hardware, edge: u32) -> Hardware {
        let square = |element| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m: edge,
            n: edge,
            k: edge,
        };
        devices::offering_cmma(hw, [square(F16), square(F32)])
    }

    /// A decode step, a prompt whose queries divide nothing, a wide group over a long cache, and
    /// a head dim of 128 in `f32`, the widest the leads are held to.
    fn problems() -> [AttentionRoutineProblem; 4] {
        [
            AttentionRoutineProblem::dense((1, 32, 8), (1, 2048), (128, 128), F16),
            AttentionRoutineProblem::dense((2, 8, 2), (37, 300), (64, 64), F16),
            AttentionRoutineProblem::dense((1, 16, 1), (512, 4096), (128, 128), F16),
            AttentionRoutineProblem::dense((1, 8, 2), (64, 256), (128, 128), F32),
        ]
    }

    /// The floor prepares every setting on every device, whatever the instructions it offers; a
    /// setting other than the default may ask for a cube the device cannot hold, and says so.
    #[test]
    fn the_floor_prepares_everywhere() {
        for hw in [
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::cpu_12(),
            offering(devices::m2_pro(), 8),
        ] {
            for (position, settings) in UnitAttentionRoutine.settings(&hw).iter().enumerate() {
                for problem in problems() {
                    let prepared = UnitAttentionRoutine.prepare(&hw, &problem, settings);
                    let outgrown = matches!(prepared, Err(Refusal::CubePastDevice { .. }));
                    assert!(
                        prepared.is_ok() || (position > 0 && outgrown),
                        "{settings} on {hw:?}: {prepared:?}"
                    );
                }
            }
        }
    }

    /// A unit reads the keys along the head dim, where they lie contiguous, with the query it    /// contracts them with; its output is lined with the values. None of them is lined along the
    /// attended keys, so each binds wider than one element.
    #[test]
    fn every_operand_binds_wider_than_one_element() {
        for hw in [devices::rdna_range(), devices::cpu_12()] {
            for problem in problems() {
                let (blueprint, _) = UnitAttentionRoutine
                    .prepare(&hw, &problem, &UnitAttentionRoutineSettings::DEFAULT)
                    .expect("the floor prepares everywhere");
                let widths = blueprint.widths;
                assert!(widths.key > 1, "{hw:?}: {widths:?}");
                assert_eq!(widths.key, widths.query);
                assert!(widths.value > 1, "{hw:?}: {widths:?}");
                assert_eq!(widths.value, widths.out);
            }
        }
    }

    /// A decode step's heads each hold one query row, so a cube's planes take rows of the next
    /// heads rather than past the last query: a cube on every core of a CPU spawns as few cubes as
    /// the heads fill.
    #[test]
    fn a_decode_step_fills_a_cube_with_heads() {
        let hw = devices::cpu_12();
        let full = UnitAttentionRoutine
            .settings(&hw)
            .into_iter()
            .find(|settings| settings.planes.get() == 12)
            .expect("a cube on every core is listed");
        let problem = AttentionRoutineProblem::dense((1, 16, 8), (1, 512), (128, 128), F16);
        let (_, grid) = UnitAttentionRoutine
            .prepare(&hw, &problem, &full)
            .expect("the floor prepares everywhere");
        assert_eq!(grid.planes().count, 12);
        assert_eq!(grid.cubes(), Some(2));
    }
}
