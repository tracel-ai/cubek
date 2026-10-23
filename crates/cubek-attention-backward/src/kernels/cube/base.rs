use super::kernel::{cube_staged_key_gradient_kernel, cube_staged_query_gradient_kernel};
use crate::{
    AttentionBackwardBlueprint, AttentionBackwardProblem, AttentionBackwardRoutine,
    AttentionBackwardRoutineBlueprint, AttentionBackwardTensors, AttentionBackwardWalk,
};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Partitioning, Space, StageStorage};
use cubek_microkernel::attention::{B, D, DV, G, KV, Q, S};
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::query_gradient::cube::{
    CubeQueryGradientMicrokernel, CubeQueryGradientMicrokernelSettings,
};
use cubek_microkernel::staging::{StageRing, StagedOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};

/// The backward pass with every cube owning, in the first walk, a box of queries of some heads of
/// the group, and in the second, a box of keys of one key head, its planes each on a matrix
/// instruction. A cube stages what its walk holds still once, and each stage of what it walks
/// once, for all its planes, met on `sync_cube`; each plane holds its own windows.
///
/// What the plane routine's planes each read again, the cube reads once: the keys and values every
/// head and every plane along the queries read in the first walk, and the queries every plane
/// along the keys reads in the second.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedAttentionBackwardRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedAttentionBackwardRoutineSettings {
    pub query_gradient: CubeQueryGradientMicrokernelSettings,
    pub key_gradient: CubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
}

impl CubeStagedAttentionBackwardRoutineSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: CubeQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: CubeKeyGradientMicrokernelSettings::DEFAULT,
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for CubeStagedAttentionBackwardRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-attention-backward query-gradient[{}] key-gradient[{}]{}",
            self.query_gradient, self.key_gradient, self.ring
        )
    }
}

impl CubeStagedAttentionBackwardRoutine {
    /// One stage of the operands a walk holds `extent` of along `walked`, a row of each spanning
    /// `D` and `DV`: the keys and values the first walk stages, or the queries and output
    /// gradient the second does.
    fn stage(
        element: ElemType,
        (walked, extent): (Axis, usize),
        problem: &AttentionBackwardProblem,
    ) -> [StagedOperand; 2] {
        let operand = |row: Axis, width: usize| StagedOperand {
            element,
            space: Space::new(&[(walked, extent), (row, width)]),
            innermost: row,
        };
        [operand(D, problem.head_dim), operand(DV, problem.val_dim)]
    }

    /// The grid and the resolved ring of a walk whose stage is `stage`, filled by `planes` planes
    /// of a cube that holds `window` bytes beside its stages.
    fn grid(
        hw: &Hardware,
        partitioning: &Partitioning,
        stage: &[StagedOperand; 2],
        ring: StageRing,
        planes: usize,
        window: usize,
    ) -> Result<(Grid, StageRing), Refusal> {
        let plane_width = hw.plane_dim()?;
        let ring = ring.resolve(hw, stage, planes * plane_width as usize);
        let stage_bytes = stage
            .iter()
            .map(|operand| operand.bytes(hw, &StageStorage::Strided))
            .sum();
        let cube = CubeResources {
            plane_width,
            shared_memory: ring.bytes(stage_bytes) + window,
        };
        Ok((Grid::new(hw, partitioning, cube)?, ring))
    }
}

impl AttentionBackwardRoutine for CubeStagedAttentionBackwardRoutine {
    type Settings = CubeStagedAttentionBackwardRoutineSettings;
    type QueryGradient = CubeQueryGradientMicrokernel;
    type KeyGradient = CubeKeyGradientMicrokernel;
    /// The settings' ring as this device holds a cube's stages.
    type SharedMemory = StageRing;

    /// Each walk's microkernel's settings, the other's and the ring at their default, then each
    /// ring of stages through registers.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = CubeStagedAttentionBackwardRoutineSettings::DEFAULT;
        let queries = CubeQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|query_gradient| Self::Settings {
                query_gradient,
                ..lead
            });
        let keys = CubeKeyGradientMicrokernel::settings(hw)
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
            .chain(rings)
            .collect()
    }

    /// A plane is as many units as the device issues its matrix instruction on; a cube walks its
    /// stages above its planes, so they share them. The first walk's cubes take the queries of
    /// some heads of every group, the second's the keys of every key head.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardRoutineBlueprint<Self>, Refusal> {
        let item = problem.work_item();
        let query_gradient =
            CubeQueryGradientMicrokernel::best(hw, &item, &settings.query_gradient)?;
        let key_gradient = CubeKeyGradientMicrokernel::best(hw, &item, &settings.key_gradient)?;
        let query_partitioning = Partitioning::new(
            problem.space(),
            query_gradient
                .tiling()
                .cubes(&[Q, G])
                .batches(&[B, KV])
                .build(),
        );
        let key_partitioning = Partitioning::new(
            problem.space(),
            key_gradient.tiling().cubes(&[S]).batches(&[B, KV]).build(),
        );
        let (query_grid, query_ring) = Self::grid(
            hw,
            &query_partitioning,
            &Self::stage(item.element, (S, query_gradient.seq_kv()), problem),
            settings.ring,
            query_gradient.plane_count(),
            query_gradient.window_bytes(&item),
        )?;
        let (key_grid, key_ring) = Self::grid(
            hw,
            &key_partitioning,
            &Self::stage(item.element, (Q, key_gradient.seq_q()), problem),
            settings.ring,
            key_gradient.plane_count(),
            key_gradient.window_bytes(&item),
        )?;
        Ok(AttentionBackwardBlueprint {
            query_gradient: AttentionBackwardWalk::new(
                hw,
                problem,
                query_gradient,
                query_partitioning,
                query_grid,
                query_ring,
            ),
            key_gradient: AttentionBackwardWalk::new(
                hw,
                problem,
                key_gradient,
                key_partitioning,
                key_grid,
                key_ring,
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
        let walk = blueprint.query_gradient;
        let binding = tensors.bind(&walk);
        let operands = binding.operands;
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' gradient and dot products, and each
        // cube's shared memory is its own.
        unsafe {
            cube_staged_query_gradient_kernel::launch_unchecked(
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
                walk.shared_memory,
                problem.visibility,
                elements,
            );
        }
        let walk = blueprint.key_gradient;
        let binding = tensors.bind(&walk);
        let operands = binding.operands;
        // SAFETY: as the first walk's; each plane writes only its own keys' and values' gradients,
        // and reads the dot products the first walk wrote, which the launches' order sees done.
        unsafe {
            cube_staged_key_gradient_kernel::launch_unchecked(
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
                walk.shared_memory,
                problem.visibility,
                elements,
            );
        }
    }
}
