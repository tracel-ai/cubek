use super::kernel::{
    cast_key_gradient_kernel, cube_staged_key_gradient_kernel, cube_staged_key_gradient_sum_kernel,
    cube_staged_query_gradient_kernel,
};
use crate::kernels::{
    AttentionBackwardBlueprint, AttentionBackwardProblem, AttentionBackwardRoutine,
    AttentionBackwardRoutineBlueprint, AttentionBackwardTensors, AttentionBackwardWalk, WalkStages,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::ComputeScope;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::query_gradient::cube::{
    CubeQueryGradientMicrokernel, CubeQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

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
    /// How the key walk's cubes take the heads of a group.
    pub key_heads: KeyHeads,
}

impl CubeStagedAttentionBackwardRoutineSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: CubeQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: CubeKeyGradientMicrokernelSettings::DEFAULT,
        ring: StageRing::SINGLE,
        key_heads: KeyHeads::Together,
    };
}

/// What heads apart decline: a group of one head has none to set apart.
const ONE_HEAD_A_GROUP: Refusal = Refusal::Declined {
    why: "the key walk's heads apart need a group of more than one head",
};

/// How the key walk's cubes take the heads of a group, which all read the same keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyHeads {
    /// Each cube walks every head of the group in turn and writes its keys' gradients.
    Together,
    /// Each cube walks one head and adds its part of the keys' gradients into sums every cube of
    /// the group adds into, cast once they are done. Under a causal mask the cube with the first
    /// keys walks every query of every head; apart, its heads run side by side.
    Apart,
}

/// Nothing for heads together, so the settings that hold it keep their names.
impl core::fmt::Display for KeyHeads {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Together => Ok(()),
            Self::Apart => write!(f, " key_heads=apart"),
        }
    }
}

impl core::fmt::Display for CubeStagedAttentionBackwardRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-attention-backward query-gradient[{}] key-gradient[{}]{}{}",
            self.query_gradient, self.key_gradient, self.ring, self.key_heads
        )
    }
}

impl AttentionBackwardRoutine for CubeStagedAttentionBackwardRoutine {
    type Settings = CubeStagedAttentionBackwardRoutineSettings;
    type QueryMicrokernel = CubeQueryGradientMicrokernel;
    type KeyMicrokernel = CubeKeyGradientMicrokernel;
    /// The settings' ring as this device holds a cube's stages.
    type SharedMemory = StageRing;

    /// Each walk's microkernel's settings, the other's and the ring at their default, then each
    /// ring of stages through registers, then each held key walk on each of them, then both walks
    /// held, then some of those with the key walk's heads apart.
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
        // A key walk holding its score in fragments runs nothing but its products between two
        // fills, so its next stage lands under them: each held setting on each ring.
        let held = CubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .filter(|key_gradient| key_gradient.holds_its_score())
            .flat_map(|key_gradient| {
                [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS].map(|ring| {
                    Self::Settings {
                        key_gradient,
                        ring,
                        ..lead
                    }
                })
            });
        // Both walks holding their scores in fragments, the key walk on up to four planes, on a
        // ring of one stage and on the one prefetching through registers.
        let query_held: Vec<_> = CubeQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .filter(|query_gradient| query_gradient.holds_its_score())
            .collect();
        let both = CubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .filter(|key_gradient| {
                key_gradient.holds_its_score() && key_gradient.planes_along_keys.get() <= 4
            })
            .flat_map(move |key_gradient| {
                query_held
                    .clone()
                    .into_iter()
                    .flat_map(move |query_gradient| {
                        [StageRing::SINGLE, StageRing::SINGLE_THROUGH_REGISTERS].map(|ring| {
                            Self::Settings {
                                query_gradient,
                                key_gradient,
                                ring,
                                ..lead
                            }
                        })
                    })
            })
            .collect::<Vec<_>>();
        // The key walk's heads apart, on the settings of both walks held whose key walk prefetches
        // through registers: where the first keys' cube, walking every head, is what the launch
        // waits on.
        let apart = both
            .iter()
            .filter(|settings| settings.ring == StageRing::SINGLE_THROUGH_REGISTERS)
            .map(|settings| Self::Settings {
                key_heads: KeyHeads::Apart,
                ..*settings
            })
            .collect::<Vec<_>>();
        core::iter::once(lead)
            .chain(queries)
            .chain(keys)
            .chain(rings)
            .chain(held)
            .chain(both)
            .chain(apart)
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
        // Under a causal mask a later box of queries sees more keys: the heaviest start first and
        // the light ones fill in behind them.
        let query_cubes = query_gradient.tiling().cubes(&[Q, G]);
        let query_cubes = match problem.visibility.is_causal() {
            true => query_cubes.reversed(Q),
            false => query_cubes,
        };
        let query_partitioning =
            Partitioning::new(problem.space(), query_cubes.batches(&[B, KV]).build());
        let key_cubes = match settings.key_heads {
            KeyHeads::Together => key_gradient.tiling().cubes(&[S]),
            KeyHeads::Apart if problem.group > 1 => key_gradient
                .tiling()
                .cubes(&[S, G])
                .across(G, problem.group),
            KeyHeads::Apart => return Err(ONE_HEAD_A_GROUP),
        };
        let key_partitioning =
            Partitioning::new(problem.space(), key_cubes.batches(&[B, KV]).build());
        let (query_grid, query_ring) = WalkStages {
            stage: problem.stage(S, query_gradient.seq_kv()),
            ring: settings.ring,
            filling_planes: query_gradient.plane_count(),
            copies: 1,
            window_bytes: query_gradient.window_bytes(&item),
            body_bytes: 0,
        }
        .grid(hw, &query_partitioning)?;
        let (key_grid, key_ring) = WalkStages {
            stage: problem.stage(Q, key_gradient.seq_q()),
            ring: settings.ring,
            filling_planes: key_gradient.plane_count(),
            copies: 1,
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
                query_ring,
            ),
            key_walk: AttentionBackwardWalk::new(
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
        let walk = blueprint.query_walk;
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

        let walk = blueprint.key_walk;
        let binding = tensors.bind(&walk);
        // A cube holding fewer heads than the group's shares its keys with the group's other
        // cubes: they add into sums, cast once every cube is done.
        let apart = walk.partitioning.box_of(ComputeScope::Cube).extent(G) < problem.group;
        if !apart {
            let operands = binding.operands;
            // SAFETY: as the first walk's; each plane writes only its own keys' and values'
            // gradients, and reads the dot products the first walk wrote, which the launches'
            // order sees done.
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
            return;
        }

        // Every cube of a group adds into them, so they start at zero.
        let client = tensors.client();
        let sum = |width: usize| {
            PlainTensor::of_handle(TensorHandle::zeros(
                client,
                [problem.batch, problem.num_kv_heads, problem.seq_kv, width],
                AttentionBackwardProblem::SUM,
            ))
        };
        let (key_sum, value_sum) = (sum(problem.head_dim), sum(problem.val_dim));
        let bind_sum =
            |tensor: &PlainTensor, operand| tensors.bind_beside(&binding, &walk, tensor, operand);
        let (key_summed, value_summed) = (
            bind_sum(&key_sum, AttentionBackwardOperand::KeyGradient),
            bind_sum(&value_sum, AttentionBackwardOperand::ValueGradient),
        );
        let operands = binding.operands;

        // SAFETY: as the first walk's; each plane only adds into its own keys' and values' sums,
        // atomically, and reads the dot products the first walk wrote, which the launches' order
        // sees done.
        unsafe {
            cube_staged_key_gradient_sum_kernel::launch_unchecked(
                client,
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                operands.query.arg(),
                operands.key.arg(),
                operands.value.arg(),
                operands.out_gradient.arg(),
                operands.logsumexp.arg(),
                operands.delta.arg(),
                key_summed.atomic::<f32>(),
                value_summed.atomic::<f32>(),
                queries,
                keys,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                walk.microkernel,
                walk.shared_memory,
                problem.visibility,
                problem.element,
            );
        }

        let cast_binding = tensors.bind(&walk);
        let bind_cast = |tensor: &PlainTensor, operand| {
            tensors.bind_beside(&cast_binding, &walk, tensor, operand)
        };
        let (key_sum, value_sum) = (
            bind_cast(&key_sum, AttentionBackwardOperand::KeyGradient),
            bind_cast(&value_sum, AttentionBackwardOperand::ValueGradient),
        );
        let cast_operands = cast_binding.operands;

        // SAFETY: as the first walk's; each plane of the cubes on a group's first head writes only
        // its own keys' and values' gradients, out of the sums the key walk finished adding into,
        // which the launches' order sees done.
        unsafe {
            cast_key_gradient_kernel::launch_unchecked(
                client,
                walk.grid.launch.cube_count,
                walk.grid.launch.cube_dim,
                key_sum.arg(),
                value_sum.arg(),
                cast_operands.key_gradient.arg(),
                cast_operands.value_gradient.arg(),
                cast_binding.launcher.partitioning_arg(),
            );
        }
    }
}
