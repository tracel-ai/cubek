use super::kernel::{
    cast_query_gradient_kernel, hypercube_key_gradient_kernel, plane_delta_kernel,
};
use crate::kernels::{
    AttentionBackwardBlueprint, AttentionBackwardProblem, AttentionBackwardRoutine,
    AttentionBackwardRoutineBlueprint, AttentionBackwardTensors, AttentionBackwardWalk,
};
use cubecl::features::AtomicUsage;
use cubecl::ir::Type;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Partitioning, StageStorage};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_microkernel::attention_backward::delta::plane::{
    PlaneDeltaMicrokernel, PlaneDeltaMicrokernelSettings,
};
use cubek_microkernel::attention_backward::key_gradient::hypercube::{
    HypercubeKeyGradientMicrokernel, HypercubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::microkernel::{StageRing, StagedOperand};
use cubek_std::routine::{CubeResources, Grid};

/// What the hypercube routine declines on a device that cannot add `f32` atomically: the query's
/// gradient is summed across the cubes that own its keys.
pub const ATTENTION_BACKWARD_NO_ATOMIC_ADD: Refusal = Refusal::Declined {
    why: "the hypercube backward pass sums the query's gradient with f32 atomic adds, which this \
          device does not have",
};

/// The backward pass with one walk owned by the keys, which the hypercube scope extends to the
/// query's gradient: every cube owns a box of keys of one key head as the cube staged routine's
/// second walk does, and also adds its share of the gradient of each query it sees into one `f32`
/// sum every such cube adds into. Five multiplications a tile instead of seven.
///
/// The first walk only writes each query's dot product of the output with its gradient, a plane a
/// box of queries; the last casts the sum into the query's gradient under the same planes. The atomic adds land in no set order,
/// so the query's gradient may differ from one run to the next in its last bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeAttentionBackwardRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAttentionBackwardRoutineSettings {
    pub delta: PlaneDeltaMicrokernelSettings,
    pub key_gradient: HypercubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
}

impl HypercubeAttentionBackwardRoutineSettings {
    pub const DEFAULT: Self = Self {
        delta: PlaneDeltaMicrokernelSettings::DEFAULT,
        key_gradient: HypercubeKeyGradientMicrokernelSettings::DEFAULT,
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for HypercubeAttentionBackwardRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-attention-backward delta[{}] key-gradient[{}]{}",
            self.delta, self.key_gradient, self.ring
        )
    }
}

impl AttentionBackwardRoutine for HypercubeAttentionBackwardRoutine {
    type Settings = HypercubeAttentionBackwardRoutineSettings;
    type QueryGradient = PlaneDeltaMicrokernel;
    type KeyGradient = HypercubeKeyGradientMicrokernel;
    /// The settings' ring as this device holds a cube's stages; the first walk holds none.
    type SharedMemory = StageRing;

    /// The key walk's microkernel's settings at the default ring, then each ring of stages through
    /// registers.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = HypercubeAttentionBackwardRoutineSettings::DEFAULT;
        let keys = HypercubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let rings = [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS]
            .map(|ring| Self::Settings { ring, ..lead });
        core::iter::once(lead).chain(keys).chain(rings).collect()
    }

    /// The first walk's planes take the queries of every head, a plane a cube; the second's cubes
    /// the keys of every key head, as the cube staged routine places them.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_NO_ATOMIC_ADD`] on a device without `f32` atomic adds, and what either
    /// microkernel or the grid declines.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardRoutineBlueprint<Self>, Refusal> {
        let atomic_sum = Type::atomic(AttentionBackwardProblem::SUM);
        if !hw
            .properties
            .atomic_type_usage(atomic_sum)
            .contains(AtomicUsage::Add)
        {
            return Err(ATTENTION_BACKWARD_NO_ATOMIC_ADD);
        }
        let item = problem.work_item();
        let delta = PlaneDeltaMicrokernel::best(hw, &item, &settings.delta)?;
        let key_gradient =
            HypercubeKeyGradientMicrokernel::best(hw, &item, &settings.key_gradient)?;
        let plane_width = hw.plane_dim()?;
        let delta_partitioning = Partitioning::new(
            problem.space(),
            delta
                .tiling()
                .launch_planes(
                    &[Q, G, B, KV],
                    PlaneOrder::new(&[(Q, 1)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let delta_grid = Grid::new(
            hw,
            &delta_partitioning,
            CubeResources {
                plane_width,
                shared_memory: 0,
            },
        )?;
        let key_partitioning = Partitioning::new(
            problem.space(),
            key_gradient.tiling().cubes(&[S]).batches(&[B, KV]).build(),
        );
        let stage = problem.stage(Q, key_gradient.seq_q());
        let ring = settings.ring.resolve(
            hw,
            &stage,
            key_gradient.plane_count() * plane_width as usize,
        );
        let key_grid = Grid::new(
            hw,
            &key_partitioning,
            CubeResources {
                plane_width,
                shared_memory: ring.bytes(Self::stage_bytes(hw, &stage))
                    + key_gradient.window_bytes(&item),
            },
        )?;
        let mut queries =
            AttentionBackwardWalk::new(hw, problem, delta, delta_partitioning, delta_grid, ring);
        // The last launch casts the sum into the query's gradient under this walk, both bound at
        // one width, which the sum's element must allow too.
        let sum_ceiling = hw.vector_ceiling(AttentionBackwardProblem::SUM);
        queries.widths = queries.widths.map(|operand, &width| match operand {
            AttentionBackwardOperand::QueryGradient => width.min(sum_ceiling),
            _ => width,
        });
        Ok(AttentionBackwardBlueprint {
            query_gradient: queries,
            key_gradient: AttentionBackwardWalk::new(
                hw,
                problem,
                key_gradient,
                key_partitioning,
                key_grid,
                ring,
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
        let client = tensors.client();
        let gradient_shape = [
            problem.batch,
            problem.num_kv_heads * problem.group,
            problem.seq_q,
            problem.head_dim,
        ];
        // Every cube adds into it, so it starts at zero.
        let sum = PlainTensor::of_handle(TensorHandle::zeros(
            client,
            gradient_shape,
            AttentionBackwardProblem::SUM,
        ));
        let walk = &blueprint.query_gradient;
        let binding = tensors.bind(walk);
        let operands = binding.operands;
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Each plane writes only its own queries' dot products.
        unsafe {
            plane_delta_kernel::launch_unchecked(
                client,
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                operands.out.arg(),
                operands.out_gradient.arg(),
                operands.delta.arg(),
                binding.launcher.partitioning_arg(),
                AttentionBackwardProblem::SUM,
            );
        }
        let elements = [AttentionBackwardProblem::SUM, problem.element];
        let key_walk = blueprint.key_gradient;
        let key_binding = tensors.bind(&key_walk);
        let summed = tensors.bind_beside(
            &key_binding,
            &key_walk,
            &sum,
            AttentionBackwardOperand::QueryGradient,
        );
        let key_operands = key_binding.operands;
        // SAFETY: as the first walk's; each plane writes only its own keys' and values' gradients,
        // and only adds into the query's gradient's sum, atomically. It reads the dot products the
        // first walk wrote, which the launches' order sees done.
        unsafe {
            hypercube_key_gradient_kernel::launch_unchecked(
                client,
                key_walk.grid.launch.cube_count,
                key_walk.grid.launch.cube_dim,
                key_operands.query.arg(),
                key_operands.key.arg(),
                key_operands.value.arg(),
                key_operands.out_gradient.arg(),
                key_operands.logsumexp.arg(),
                key_operands.delta.arg(),
                key_operands.key_gradient.arg(),
                key_operands.value_gradient.arg(),
                summed.atomic::<f32>(),
                queries,
                keys,
                tensors.scale(),
                key_binding.launcher.partitioning_arg(),
                key_walk.microkernel,
                key_walk.shared_memory,
                problem.visibility,
                elements,
            );
        }
        let cast_binding = tensors.bind(walk);
        let sum = tensors.bind_beside(
            &cast_binding,
            walk,
            &sum,
            AttentionBackwardOperand::QueryGradient,
        );
        // SAFETY: as the first walk's; each plane writes only its own queries' gradient, out of the
        // sum the key walk finished adding into, which the launches' order sees done.
        unsafe {
            cast_query_gradient_kernel::launch_unchecked(
                client,
                walk.grid.launch.cube_count.clone(),
                walk.grid.launch.cube_dim,
                sum.arg(),
                cast_binding.operands.query_gradient.arg(),
                cast_binding.launcher.partitioning_arg(),
            );
        }
    }
}

impl HypercubeAttentionBackwardRoutine {
    /// Bytes one stage of `stage` takes, strided.
    fn stage_bytes(hw: &Hardware, stage: &[StagedOperand; 2]) -> usize {
        stage
            .iter()
            .map(|operand| operand.bytes(hw, &StageStorage::Strided))
            .sum()
    }
}
