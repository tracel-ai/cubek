use super::kernel::plane_fragment_attention_kernel;
use crate::kernels::{
    ATTENTION_PACKED_UNREAD, AttentionBinding, AttentionBlueprint, AttentionCacheLayout,
    AttentionRoutine, AttentionRoutineProblem, AttentionTensors,
};
use core::num::NonZeroUsize;
use cubecl::tile::StageStorage;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::attention::plane::{
    InstructionShape, PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, AttentionPartitioning,
    AttentionStagingCount, CubePlaneMicrokernel, Q, S,
};
use cubek_microkernel::staging::StageRing;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with every plane of the launch owning a box of rows of one head, and nothing else:
/// it stages its own rows of the query once, and its keys and values a stage at a time, in a
/// window of its own filled by its own units and met on `sync_plane`; no plane waits on another.
///
/// The launch's planes cover the queries and the heads, one batch and key head each, and the
/// launch groups them in cubes along the queries of one head. What a plane does not share it reads
/// again: every plane of a cube, and every cube of the group, stages the same keys and values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentAttentionRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionRoutineSettings {
    pub microkernel: PlaneFragmentAttentionMicrokernelSettings,
    /// Planes the launch groups in one cube, along the queries.
    pub planes: NonZeroUsize,
    /// How a plane holds its stages of keys and values.
    pub ring: StageRing,
}

impl PlaneFragmentAttentionRoutineSettings {
    /// One plane a cube: what holds a head dim of 128 in `f32` within 32 KiB of shared memory.
    pub const DEFAULT: Self = Self {
        microkernel: PlaneFragmentAttentionMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(1).unwrap(),
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for PlaneFragmentAttentionRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "plane-fragment-attention {} planes={}{}",
            self.microkernel, self.planes, self.ring
        )
    }
}

/// How the plane routine lays out the shared memory its walk holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionRoutineSharedMemory {
    /// The settings' ring as this device holds a plane's stages ([`StageRing::resolve`]).
    pub ring: StageRing,
    /// The planes of a cube, each holding its own windows.
    pub planes: usize,
}

pub type PlaneFragmentAttentionRoutineBlueprint = AttentionBlueprint<
    PlaneFragmentAttentionMicrokernel,
    PlaneFragmentAttentionRoutineSharedMemory,
>;

impl PlaneFragmentAttentionRoutine {
    /// Bytes one plane's stages hold on `hw`: its rows of the query once, and each stage `ring`
    /// holds of keys and values.
    fn stage_bytes(
        hw: &Hardware,
        microkernel: &PlaneFragmentAttentionMicrokernel,
        problem: &AttentionRoutineProblem,
        ring: &StageRing,
    ) -> usize {
        let query = microkernel.seq_q() * problem.head_dim * problem.elements.query.size();
        let stage: usize = problem
            .work_item()
            .stage(microkernel.seq_kv(), microkernel.key_order())
            .iter()
            .map(|operand| operand.bytes(hw, &StageStorage::Strided))
            .sum();
        query + ring.bytes(stage)
    }
}

impl Routine for PlaneFragmentAttentionRoutine {
    type Problem = AttentionRoutineProblem;
    type Microkernel = PlaneFragmentAttentionMicrokernel;
    type Settings = PlaneFragmentAttentionRoutineSettings;
    type Blueprint = PlaneFragmentAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// A star off the default: its microkernel's boxes of fragments, each count of planes, each
    /// other ring of stages, then its microkernel's whole grid bounced at once and the score read
    /// through the windows on a square and on a rectangular cmma instruction — the vendor's
    /// transport beside the lead, so the race still weighs the windows on a device that offers
    /// both — each with the default's others. Every microkernel setting is one its microkernel
    /// lists ([`PlaneFragmentAttentionMicrokernel::settings`]).
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        use PlaneFragmentAttentionMicrokernelSettings as Plane;
        let lead = PlaneFragmentAttentionRoutineSettings::DEFAULT;
        let on = |microkernel| Self::Settings {
            microkernel,
            ..lead
        };
        let planes = [2, 4].map(|planes| Self::Settings {
            planes: NonZeroUsize::new(planes).unwrap(),
            ..lead
        });
        let rings = [
            StageRing::DOUBLE,
            StageRing::SINGLE_THROUGH_REGISTERS,
            StageRing::DOUBLE_THROUGH_REGISTERS,
        ]
        .map(|ring| Self::Settings { ring, ..lead });
        [
            lead,
            on(Plane::fragments(1, 1)),
            on(Plane::fragments(4, 2)),
            on(Plane::fragments(2, 4)),
        ]
        .into_iter()
        .chain(planes)
        .chain(rings)
        .chain([
            on(Plane::WHOLE_GRID),
            on(Plane::through_windows(InstructionShape::Square)),
            on(Plane::through_windows(InstructionShape::Rectangular)),
        ])
        .collect()
    }

    /// Nothing of its own: its stages copy the tiles its microkernel reads.
    fn vectorize(
        &self,
        microkernel: &PlaneFragmentAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Its planes' stages: each plane stages its rows of the query once, and every key and value of
    /// the walk, in runs of a row each; every plane of the cube reading the keys and values again.
    /// Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &PlaneFragmentAttentionRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, partitioning) = (problem.work_item(), &blueprint.partitioning);
        let microkernel = &blueprint.microkernel;
        let cube = partitioning.cube_box();
        let readers = (cube.extent(Q) / microkernel.seq_q()) as u64;
        AttentionStagingCount {
            seq_kv: microkernel.seq_kv(),
            stages_in_flight: blueprint.shared_memory.ring.stages_in_flight.get() as u32,
            runs: AttentionOperands {
                query: item.head_dim,
                key: item.head_dim,
                value: item.val_dim,
                out: item.val_dim,
            },
            readers: AttentionOperands {
                query: 1,
                key: readers,
                value: readers,
                out: 1,
            },
        }
        .count(&item, partitioning, estimate);
        microkernel.estimate(&item, partitioning, grid.planes(), estimate);
    }

    /// A plane is as many units as the device issues its matrix instruction on, and it walks its
    /// own keys below the planes' level, so each stages its own copy ([`Stages`](cubecl::tile::Stages)).
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
        let microkernel =
            PlaneFragmentAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
        let planes = settings.planes.get();
        let partitioning = AttentionPartitioning::over_planes(
            problem.space(),
            microkernel.tiling().walk_every(&[S]),
            PlaneOrder::new(&[(Q, planes)], CubeOrder::RowMajor),
        );
        let plane_width = hw.plane_dim()?;
        let ring = settings.ring.resolve(
            hw,
            &microkernel
                .served(&item)
                .stage(microkernel.seq_kv(), microkernel.key_order()),
            plane_width as usize,
        );
        let cube = CubeResources {
            plane_width,
            shared_memory: planes
                * (Self::stage_bytes(hw, &microkernel, problem, &ring)
                    + microkernel.window_bytes(&item)),
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
            PlaneFragmentAttentionRoutineSharedMemory { ring, planes },
        );
        Ok((blueprint, grid))
    }
}

impl AttentionRoutine<PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionRoutineSharedMemory>
    for PlaneFragmentAttentionRoutine
{
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: PlaneFragmentAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let (key, value) = binding.cache.into_tiles();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short or a plane walks past the grid, the keys and
        // values are read no further than the attended length, and the kernel indexes no buffer
        // but through its tiles and its sequence's entry of the lengths: a plane past the grid
        // lies past the last key head, never past the last batch. Each plane's shared memory is
        // its own.
        unsafe {
            plane_fragment_attention_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                binding.query.arg(),
                key.arg(),
                value.arg(),
                binding.out.arg(),
                binding.lengths,
                problem.seq_q as u32,
                tensors.scale(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.shared_memory.ring,
                blueprint.shared_memory.planes,
                problem.visibility,
                blueprint
                    .microkernel
                    .served(&problem.work_item())
                    .kernel_elements(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::attention::plane::ScoreHolder;
    use cubek_microkernel::attention::{D, G};
    use cubek_microkernel::matmul::MatmulDim;
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentTransport;
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

    /// Every setting prepares where the device offers a square instruction and commits to a plane
    /// width; one other than the default may ask for more than a cube holds, and says so.
    #[test]
    fn every_setting_prepares_where_planes_run() {
        for hw in [
            offering(devices::m2_pro(), 8),
            offering(devices::h100(), 16),
            offering(devices::strix_halo(), 16),
        ] {
            for (position, settings) in PlaneFragmentAttentionRoutine
                .settings(&hw)
                .iter()
                .enumerate()
            {
                for problem in problems() {
                    let prepared = PlaneFragmentAttentionRoutine.prepare(&hw, &problem, settings);
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

    /// Without a matrix instruction, or a committed plane width, the routine declines with the
    /// microkernel's reason.
    #[test]
    fn declines_without_a_matrix_instruction() {
        for hw in [
            devices::wgpu_no_subgroups(),
            offering(devices::rdna_range(), 16),
        ] {
            let prepared = PlaneFragmentAttentionRoutine.prepare(
                &hw,
                &problems()[0],
                &PlaneFragmentAttentionRoutineSettings::DEFAULT,
            );
            assert!(
                matches!(
                    prepared,
                    Err(Refusal::Declined { why: details })
                        if details == PlaneFragmentAttentionMicrokernel::DECLINE
                ),
                "{hw:?}: {prepared:?}"
            );
        }
    }

    /// A rectangular instruction runs both products: its `n` cuts the score's keys and its `k`
    /// the value's, so `16×8×16` takes two of the score's fragments to one step of the value's.
    #[test]
    fn a_rectangular_instruction_runs() {
        let rectangular = devices::offering_cmma(
            devices::strix_halo(),
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: F32,
                m: 16,
                n: 8,
                k: 16,
            }],
        );
        let (blueprint, _) = PlaneFragmentAttentionRoutine
            .prepare(
                &rectangular,
                &problems()[0],
                &PlaneFragmentAttentionRoutineSettings::DEFAULT,
            )
            .expect("16×8×16 prepares");
        let microkernel = blueprint.microkernel;
        assert_eq!(
            microkernel.score.tile_extent(S),
            microkernel.value.tile_extent(S)
        );
        assert_eq!(
            microkernel.value.settings.depth.get() * 16,
            microkernel.seq_kv()
        );
    }

    /// A plane stages its own box: a stage is its rows of one head against its keys, the head dim
    /// whole, and the cube holds one copy for every plane.
    #[test]
    fn every_plane_stages_its_own_box() {
        let hw = offering(devices::m2_pro(), 8);
        let (blueprint, grid) = PlaneFragmentAttentionRoutine
            .prepare(
                &hw,
                &problems()[1],
                &PlaneFragmentAttentionRoutineSettings::DEFAULT,
            )
            .expect("the default prepares on the M2");
        let (stage, microkernel) = (blueprint.partitioning.stage(), blueprint.microkernel);
        assert_eq!(stage.extent(Q), microkernel.seq_q());
        assert_eq!(stage.extent(G), 1);
        assert_eq!(stage.extent(S), microkernel.seq_kv());
        assert_eq!(stage.extent(D), 64);
        let planes = PlaneFragmentAttentionRoutineSettings::DEFAULT.planes.get();
        assert_eq!(grid.planes().count, planes);
    }

    /// `hw` offering, for `f16` summed in `f32`, the manual `mma` of `16 × 8 × k` for each `k`,
    /// and no cmma.
    fn offering_manual(hw: Hardware, depths: &[u32]) -> Hardware {
        let shape = |k| MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 8,
            k,
        };
        devices::offering_mma(
            devices::offering_cmma(hw, []),
            depths.iter().map(|&k| shape(k)),
        )
    }

    /// A manual `mma` beside its half-depth twin keeps the score in its fragments: the value
    /// contracts the keys as deep as the score is wide, so no window is held for either.
    #[test]
    fn keeps_the_score_in_fragments_beside_a_half_depth_mma() {
        let hw = offering_manual(devices::h100(), &[16, 8]);
        let (blueprint, _) = PlaneFragmentAttentionRoutine
            .prepare(
                &hw,
                &problems()[2],
                &PlaneFragmentAttentionRoutineSettings::DEFAULT,
            )
            .expect("the default prepares beside a manual pair");
        let microkernel = blueprint.microkernel;
        assert_eq!(microkernel.score_holder(), ScoreHolder::Fragments);
        assert_eq!(microkernel.value.leaf_extent(MatmulDim::K), 8);
        assert_eq!(
            blueprint.partitioning.stage().extent(S),
            microkernel.seq_kv()
        );
        assert_eq!(microkernel.window_bytes(&problems()[2].work_item()), 0);
    }

    /// Where a device offers both, the transport setting says where the score is kept: the tile
    /// DSL's own in the fragments, the lead; the vendor's through the windows, raced beside it.
    #[test]
    fn the_transport_says_where_the_score_is_kept() {
        let manual = offering_manual(devices::h100(), &[16, 8]);
        let hw = devices::offering_mma(
            offering(devices::h100(), 16),
            manual.properties.features.matmul.mma.iter().copied(),
        );
        let holder = |transport| {
            let settings = PlaneFragmentAttentionRoutineSettings {
                microkernel: PlaneFragmentAttentionMicrokernelSettings {
                    transport,
                    ..PlaneFragmentAttentionRoutineSettings::DEFAULT.microkernel
                },
                ..PlaneFragmentAttentionRoutineSettings::DEFAULT
            };
            let (blueprint, _) = PlaneFragmentAttentionRoutine
                .prepare(&hw, &problems()[2], &settings)
                .expect("both transports prepare where both are offered");
            blueprint.microkernel.score_holder()
        };
        assert_eq!(holder(PlaneFragmentTransport::Mma), ScoreHolder::Fragments);
        assert_eq!(holder(PlaneFragmentTransport::Cmma), ScoreHolder::Window);
    }

    /// Where the device lays a manual `mma`'s accumulator out apart from its `A` operand, as RDNA
    /// does, no score is kept in fragments: the value would read the score's cells from the wrong
    /// registers. Beside a cmma instruction the plane reads the score through its windows instead;
    /// with the manual `mma` alone it declines.
    #[test]
    fn keeps_no_score_in_fragments_laid_out_apart_from_a() {
        let square = MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m: 16,
            n: 16,
            k: 16,
        };
        let manual =
            devices::offering_mma(devices::offering_cmma(devices::strix_halo(), []), [square]);
        assert!(!manual.accumulator_as_a);
        let alone = PlaneFragmentAttentionRoutine.prepare(
            &manual,
            &problems()[2],
            &PlaneFragmentAttentionRoutineSettings::DEFAULT,
        );
        assert!(
            matches!(
                alone,
                Err(Refusal::Declined { why: details })
                    if details == PlaneFragmentAttentionMicrokernel::DECLINE
            ),
            "{alone:?}"
        );

        let beside = devices::offering_mma(
            offering(devices::strix_halo(), 16),
            manual.properties.features.matmul.mma.iter().copied(),
        );
        let (blueprint, _) = PlaneFragmentAttentionRoutine
            .prepare(
                &beside,
                &problems()[2],
                &PlaneFragmentAttentionRoutineSettings::DEFAULT,
            )
            .expect("the cmma instruction serves");
        assert_eq!(blueprint.microkernel.score_holder(), ScoreHolder::Window);
    }

    /// A `tf32` manual `mma` keeps no score in its fragments, beside its twin or not: its `A` lays
    /// its cells along the columns first, where the accumulator's run along the rows.
    #[test]
    fn keeps_no_tf32_score_in_fragments() {
        let tf32 = |k| MmaConfig {
            a_type: F32,
            b_type: F32,
            cd_type: F32,
            m: 16,
            n: 8,
            k,
        };
        let hw = devices::offering_mma(devices::offering_cmma(devices::h100(), []), [tf32(8)]);
        let prepared = PlaneFragmentAttentionRoutine.prepare(
            &hw,
            &problems()[3],
            &PlaneFragmentAttentionRoutineSettings::DEFAULT,
        );
        assert!(
            matches!(
                prepared,
                Err(Refusal::Declined { why: details })
                    if details == PlaneFragmentAttentionMicrokernel::DECLINE
            ),
            "{prepared:?}"
        );
    }

    /// A manual `mma` whose half-depth twin the device does not offer has no instruction for the
    /// value to read the score's fragments through, and the routine declines.
    #[test]
    fn declines_a_manual_mma_without_its_half_depth_twin() {
        let hw = offering_manual(devices::h100(), &[16]);
        let prepared = PlaneFragmentAttentionRoutine.prepare(
            &hw,
            &problems()[2],
            &PlaneFragmentAttentionRoutineSettings::DEFAULT,
        );
        assert!(
            matches!(
                prepared,
                Err(Refusal::Declined { why: details })
                    if details == PlaneFragmentAttentionMicrokernel::DECLINE
            ),
            "{prepared:?}"
        );
    }
}
