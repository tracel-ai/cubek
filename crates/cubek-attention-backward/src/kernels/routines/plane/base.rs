use super::kernel::{plane_fragment_key_gradient_kernel, plane_fragment_query_gradient_kernel};
use crate::kernels::{
    AttentionBackwardBlueprint, AttentionBackwardProblem, AttentionBackwardRoutine,
    AttentionBackwardRoutineBlueprint, AttentionBackwardTensors, AttentionBackwardWalk, WalkStages,
};
use core::num::NonZeroUsize;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_microkernel::attention_backward::key_gradient::plane::{
    PlaneKeyGradientMicrokernel, PlaneKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, PlaneQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// The backward pass with every plane owning, in the first walk, a box of queries of one head, and
/// in the second, a box of keys of one key head, each on a matrix instruction. A plane stages what
/// its walk holds still once, and what it walks a stage at a time, in shared memory of its own
/// filled by its own units and met on `sync_plane`; no plane waits on another.
///
/// Each launch groups its planes in cubes along the queries, or the keys, of one head. What a
/// plane does not share it reads again: every plane of a cube stages the same keys and values in
/// the first walk, and the same queries in the second.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentAttentionBackwardRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionBackwardRoutineSettings {
    pub query_gradient: PlaneQueryGradientMicrokernelSettings,
    pub key_gradient: PlaneKeyGradientMicrokernelSettings,
    /// Planes each launch groups in one cube.
    pub planes: NonZeroUsize,
    /// How a plane holds its stages.
    pub ring: StageRing,
}

impl PlaneFragmentAttentionBackwardRoutineSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: PlaneQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: PlaneKeyGradientMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(1).unwrap(),
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for PlaneFragmentAttentionBackwardRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "plane-fragment-attention-backward query-gradient[{}] key-gradient[{}] planes={}{}",
            self.query_gradient, self.key_gradient, self.planes, self.ring
        )
    }
}

/// How a walk of the plane routine lays out the shared memory it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionBackwardSharedMemory {
    /// The settings' ring as this device holds a plane's stages ([`StageRing::resolve`]).
    pub ring: StageRing,
    /// The planes of a cube, each holding its own windows.
    pub planes: usize,
}

impl AttentionBackwardRoutine for PlaneFragmentAttentionBackwardRoutine {
    type Settings = PlaneFragmentAttentionBackwardRoutineSettings;
    type QueryMicrokernel = PlaneQueryGradientMicrokernel;
    type KeyMicrokernel = PlaneKeyGradientMicrokernel;
    type SharedMemory = PlaneFragmentAttentionBackwardSharedMemory;

    /// Each walk's microkernel's settings, the other's and the routine's at their default, then
    /// two planes a cube, and each ring of stages through registers.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = PlaneFragmentAttentionBackwardRoutineSettings::DEFAULT;
        let queries = PlaneQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|query_gradient| Self::Settings {
                query_gradient,
                ..lead
            });
        let keys = PlaneKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let rings = [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS]
            .map(|ring| Self::Settings { ring, ..lead });
        core::iter::once(lead)
            .chain(queries)
            .chain(keys)
            .chain([Self::Settings {
                planes: NonZeroUsize::new(2).unwrap(),
                ..lead
            }])
            .chain(rings)
            .collect()
    }

    /// A plane is as many units as the device issues its matrix instruction on, and it walks its
    /// own stages below the planes' level, so each holds its own copy. The first walk's planes take
    /// the queries of every head, the second's the keys of every key head.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardRoutineBlueprint<Self>, Refusal> {
        let item = problem.work_item();
        let query_gradient =
            PlaneQueryGradientMicrokernel::best(hw, &item, &settings.query_gradient)?;
        let key_gradient = PlaneKeyGradientMicrokernel::best(hw, &item, &settings.key_gradient)?;
        let planes = settings.planes.get();
        let query_partitioning = Partitioning::new(
            problem.space(),
            query_gradient
                .tiling()
                .launch_planes(
                    &[Q, G, B, KV],
                    PlaneOrder::new(&[(Q, planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let key_partitioning = Partitioning::new(
            problem.space(),
            key_gradient
                .tiling()
                .launch_planes(
                    &[S, B, KV],
                    PlaneOrder::new(&[(S, planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let (query_grid, query_ring) = WalkStages {
            stage: problem.stage(S, query_gradient.seq_kv()),
            ring: settings.ring,
            filling_planes: 1,
            copies: planes,
            window_bytes: query_gradient.window_bytes(&item),
            body_bytes: 0,
        }
        .grid(hw, &query_partitioning)?;
        let (key_grid, key_ring) = WalkStages {
            stage: problem.stage(Q, key_gradient.seq_q()),
            ring: settings.ring,
            filling_planes: 1,
            copies: planes,
            window_bytes: key_gradient.window_bytes(&item),
            body_bytes: 0,
        }
        .grid(hw, &key_partitioning)?;
        Ok(AttentionBackwardBlueprint {
            query_walk: AttentionBackwardWalk::new(
                hw,
                problem,
                query_gradient,
                query_partitioning,
                query_grid,
                PlaneFragmentAttentionBackwardSharedMemory {
                    ring: query_ring,
                    planes,
                },
            ),
            key_walk: AttentionBackwardWalk::new(
                hw,
                problem,
                key_gradient,
                key_partitioning,
                key_grid,
                PlaneFragmentAttentionBackwardSharedMemory {
                    ring: key_ring,
                    planes,
                },
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
        let elements = [AttentionBackwardProblem::SUM, problem.element];
        let walk = blueprint.query_walk;
        let binding = tensors.bind(&walk);
        let operands = binding.operands;

        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' gradient and dot products, and its
        // shared memory is its own.
        unsafe {
            plane_fragment_query_gradient_kernel::launch_unchecked(
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
                walk.shared_memory.ring,
                walk.shared_memory.planes,
                problem.visibility,
                elements,
            );
        }

        let walk = blueprint.key_walk;
        let binding = tensors.bind(&walk);
        let operands = binding.operands;

        // SAFETY: as the first walk's; each plane writes only its own keys' and values' gradients,
        // and reads the dot products the first walk wrote, which the launches' order sees done.
        unsafe {
            plane_fragment_key_gradient_kernel::launch_unchecked(
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
                walk.shared_memory.ring,
                walk.shared_memory.planes,
                problem.visibility,
                elements,
            );
        }
    }
}
