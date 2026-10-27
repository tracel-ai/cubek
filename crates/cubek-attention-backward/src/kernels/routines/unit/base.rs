use super::kernel::{unit_key_gradient_kernel, unit_query_gradient_kernel};
use crate::kernels::{
    AttentionBackwardBlueprint, AttentionBackwardProblem, AttentionBackwardRoutine,
    AttentionBackwardRoutineBlueprint, AttentionBackwardTensors, AttentionBackwardWalk,
};
use core::num::NonZeroUsize;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::UnitCube;
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_microkernel::attention_backward::key_gradient::unit::{
    UnitKeyGradientMicrokernel, UnitKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::query_gradient::unit::{
    UnitQueryGradientMicrokernel, UnitQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};

/// The backward pass with every unit owning, in the first walk, queries of one head, and in the
/// second, keys of one key head. This kind's floor: it asks nothing of the hardware, so it runs
/// everywhere.
///
/// Each walk's units cover the problem, one batch and key head each, the queries or the keys
/// taken in turn and on into the next head. Every unit reads its operands where they lie and
/// holds nothing in shared memory: no unit waits on another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitAttentionBackwardRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitAttentionBackwardRoutineSettings {
    pub query_gradient: UnitQueryGradientMicrokernelSettings,
    pub key_gradient: UnitKeyGradientMicrokernelSettings,
    /// Planes each launch groups its units in, a cube.
    pub planes: NonZeroUsize,
}

impl UnitAttentionBackwardRoutineSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: UnitQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: UnitKeyGradientMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(2).unwrap(),
    };
}

impl core::fmt::Display for UnitAttentionBackwardRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-attention-backward query-gradient[{}] key-gradient[{}] planes={}",
            self.query_gradient, self.key_gradient, self.planes
        )
    }
}

impl AttentionBackwardRoutine for UnitAttentionBackwardRoutine {
    type Settings = UnitAttentionBackwardRoutineSettings;
    type QueryGradient = UnitQueryGradientMicrokernel;
    type KeyGradient = UnitKeyGradientMicrokernel;
    type SharedMemory = ();

    /// Each walk's microkernel's settings, the other's and the routine's at their default, then a
    /// cube as full as the device holds ([`Hardware::full_cube_planes`]).
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = UnitAttentionBackwardRoutineSettings::DEFAULT;
        let queries = UnitQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|query_gradient| Self::Settings {
                query_gradient,
                ..lead
            });
        let keys = UnitKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let full = Self::Settings {
            planes: NonZeroUsize::new(hw.full_cube_planes()).unwrap(),
            ..lead
        };
        core::iter::once(lead)
            .chain(queries)
            .chain(keys)
            .chain([full])
            .collect()
    }

    /// The units cooperate on nothing, so a cube groups its settings' planes, each as wide as the
    /// device dispatches one and no wider than the cube's unit budget leaves for each: any width
    /// is correct. The first walk's units take the queries of every head, the second's the keys of
    /// every key head.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardRoutineBlueprint<Self>, Refusal> {
        let item = problem.work_item();
        let query_gradient =
            UnitQueryGradientMicrokernel::best(hw, &item, &settings.query_gradient)?;
        let key_gradient = UnitKeyGradientMicrokernel::best(hw, &item, &settings.key_gradient)?;
        let planes = settings.planes.get();
        let plane_width = (hw.properties.hardware.max_units_per_cube / planes as u32)
            .min(hw.dispatch_plane_size())
            .max(1);
        let cube = UnitCube {
            plane_width: plane_width as usize,
            planes,
        };
        let resources = CubeResources {
            plane_width,
            shared_memory: 0,
        };
        let query_partitioning = Partitioning::new(
            problem.space(),
            query_gradient
                .tiling()
                .launch_units(&[Q, G, B, KV], cube)
                .build(),
        );
        let key_partitioning = Partitioning::new(
            problem.space(),
            key_gradient
                .tiling()
                .launch_units(&[S, B, KV], cube)
                .build(),
        );
        let query_grid = Grid::new(hw, &query_partitioning, resources)?;
        let key_grid = Grid::new(hw, &key_partitioning, resources)?;
        Ok(AttentionBackwardBlueprint {
            query_gradient: AttentionBackwardWalk::new(
                hw,
                problem,
                query_gradient,
                query_partitioning,
                query_grid,
                (),
            ),
            key_gradient: AttentionBackwardWalk::new(
                hw,
                problem,
                key_gradient,
                key_partitioning,
                key_grid,
                (),
            ),
        })
    }

    fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        blueprint: AttentionBackwardRoutineBlueprint<Self>,
    ) {
        let problem = tensors.problem();
        let (queries, keys) = (problem.seq_q as u32, problem.seq_kv as u32);
        let sum = AttentionBackwardProblem::SUM;
        let walk = blueprint.query_gradient;
        let binding = tensors.bind(&walk);
        let operands = binding.operands;
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each unit writes only its own queries' gradient and dot products.
        unsafe {
            unit_query_gradient_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count,
                walk.grid.launch.cube_dim,
                operands.query.arg(),
                operands.key.arg(),
                operands.value.arg(),
                operands.out.arg(),
                operands.out_gradient.arg(),
                operands.logsumexp.arg(),
                operands.delta.arg(),
                operands.query_gradient.arg(),
                queries,
                keys,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                walk.microkernel,
                problem.visibility,
                sum,
            );
        }
        let walk = blueprint.key_gradient;
        let binding = tensors.bind(&walk);
        let operands = binding.operands;
        // SAFETY: as the first walk's; each unit writes only its own keys' and values' gradients,
        // and reads the dot products the first walk wrote, which the launches' order sees done.
        unsafe {
            unit_key_gradient_kernel::launch_unchecked(
                tensors.client(),
                walk.grid.launch.cube_count,
                walk.grid.launch.cube_dim,
                operands.query.arg(),
                operands.key.arg(),
                operands.value.arg(),
                operands.out_gradient.arg(),
                operands.logsumexp.arg(),
                operands.delta.arg(),
                operands.key_gradient.arg(),
                operands.value_gradient.arg(),
                queries,
                keys,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                walk.microkernel,
                problem.visibility,
                sum,
            );
        }
    }
}
