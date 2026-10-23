use super::kernel::plane_fragment_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{CUBE_STRIPS, cube_order_name};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::MatmulPartitioning;
use cubek_microkernel::matmul::{MatmulAxes, MatmulDestination, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;

use crate::kernels::{
    MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Scratch, StageStorage, TileArgLaunch, launch::DestinationLaunch};
use cubek_microkernel::matmul::dense::plane::{
    PlaneFragmentDrain, PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use cubek_microkernel::matmul::staged_walk::{MatmulPlaneLayout, MatmulStagingCount};
use cubek_microkernel::staging::StageRing;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every plane owning a tile of the output from loading to writing: it
/// stages its own tile's rows of the lhs and columns of the rhs a step of `K` at a time, in shared
/// memory of its own behind its own barriers, contracts them with the plane microkernel, and
/// drains the tile. The planes share nothing but the launch: they walk the launch's planes, and
/// a cube only packs them.
///
/// Where the planes of a cube would read the same rows and columns, the cube routine stages them
/// once for all of them
/// ([`CubeStagedMatmulRoutine`](crate::kernels::CubeStagedMatmulRoutine)); here every
/// plane reads its own, so a box's rows are read once per plane along `n` and its columns once per
/// plane along `m`.
///
/// A fragment that bounces into the output drains through the plane's own scratch window, but
/// behind the cube's barriers, which the tile DSL's drain raises: every plane of the cube reaches
/// them, the same count each, a plane past the last tile walking one past the grid, so the planes
/// still share no work, only the drain's rendezvous.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentMatmulRoutineSettings {
    pub microkernel: PlaneFragmentMicrokernelSettings,
    /// The planes one cube packs, laid out over its box: each owns its own tile of it.
    pub planes: MatmulPlaneLayout,
    /// The order the cubes take their blocks of planes in.
    pub cube_order: CubeOrder,
    /// How each plane holds its own stages.
    pub ring: StageRing,
}

impl PlaneFragmentMatmulRoutineSettings {
    /// The plane microkernel's default on each of two by two planes, one stage at a time.
    pub const DEFAULT: Self = Self {
        microkernel: PlaneFragmentMicrokernelSettings::DEFAULT,
        planes: MatmulPlaneLayout::new(2, 2),
        cube_order: CubeOrder::RowMajor,
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for PlaneFragmentMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "plane-fragment-matmul {} {}{}{}",
            self.microkernel,
            self.planes,
            self.ring,
            cube_order_name(self.cube_order)
        )
    }
}

/// How the plane routine lays out the shared memory its planes' walks hold: every plane its own
/// copy of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlaneFragmentMatmulRoutineSharedMemory {
    /// How a stage holds its cells, as the plane microkernel reads them.
    pub stage: StageStorage,
    /// How many stages each plane holds, and where the next waits.
    pub ring: StageRing,
    /// The window a fragment bounces through on its way into the output, where its own store
    /// intrinsic cannot land it; `None` where it can.
    pub scratch: Scratch,
}

pub type PlaneFragmentMatmulRoutineBlueprint =
    MatmulBlueprint<PlaneFragmentMicrokernel, PlaneFragmentMatmulRoutineSharedMemory>;

impl PlaneFragmentMatmulRoutine {
    /// Each plane's ring reads both operands out of their tiles a stage at a time, in stretches of
    /// the stage's extent along each one's contiguous axis, and its drain writes the output a
    /// fragment at a time, or a cell at a time from a manual fragment. The stage is the one the
    /// planes' level leaves a plane, so only the routine that lays the planes out can state it.
    fn require_ring_accesses(
        microkernel: &PlaneFragmentMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        partitioning
            .stage()
            .require_runs(microkernel.orders, vectorization);
        vectorization.require(VectorRequirement::Access {
            operand: MatmulOperand::Out,
            elements: partitioning
                .leaf()
                .extent(MatmulAxes::MNK.of(microkernel.orders.innermost(MatmulOperand::Out))),
        });
        if microkernel.instruction.drain() == PlaneFragmentDrain::CellByCell {
            vectorization.require(VectorRequirement::Elementwise {
                operand: MatmulOperand::Out,
            });
        }
    }

    /// Planes of a cube reading each element of each operand's box under `partitioning`: every
    /// plane along `n` reads the rows of the lhs its tile spans, every plane along `m` the columns
    /// of the rhs, and each writes its own tile of the output. A plane's tile is its stage's box
    /// over `m` and `n`.
    fn readers(partitioning: &MatmulPartitioning) -> MatmulOperands<u64> {
        let (cube, stage) = (partitioning.cube_box(), partitioning.stage());
        let planes_along = |axis| (cube.extent(axis) / stage.extent(axis)) as u64;
        MatmulOperands {
            lhs: planes_along(N),
            rhs: planes_along(M),
            out: 1,
        }
    }
}

impl Routine for PlaneFragmentMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = PlaneFragmentMicrokernel;
    type Settings = PlaneFragmentMatmulRoutineSettings;
    type Blueprint = PlaneFragmentMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's grids ([`PlaneFragmentMicrokernel::settings`]), each at the routine's
    /// default, then a star around the default over its own knobs: the planes a cube packs, a
    /// second stage in flight or the next held in registers, and the strips along `m`. No
    /// measurement has placed this routine yet, so the list is what a first one reads.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = Self::Settings::DEFAULT;
        let grids = PlaneFragmentMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                ..lead
            });
        let layouts = [(1, 1), (2, 1), (4, 2)].map(|(rows, columns)| Self::Settings {
            planes: MatmulPlaneLayout::new(rows, columns),
            ..lead
        });
        let rings = [
            StageRing::DOUBLE,
            StageRing::SINGLE_THROUGH_REGISTERS,
            StageRing::DOUBLE_THROUGH_REGISTERS,
        ]
        .map(|ring| Self::Settings { ring, ..lead });
        let [along_m, _] = CUBE_STRIPS;
        let strips = Self::Settings {
            cube_order: along_m,
            ..lead
        };
        grids.chain(layouts).chain(rings).chain([strips]).collect()
    }

    /// Each plane's ring and drain, which the routine's planes lay out, then its microkernel's.
    fn vectorize(
        &self,
        microkernel: &PlaneFragmentMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        Self::require_ring_accesses(microkernel, partitioning, vectorization);
        microkernel.vectorize(&(), vectorization);
    }

    /// Each plane's ring, staging a step of its own tile's `K` at a time: rows of the lhs
    /// contiguous along the whole of `K` a cube walks, of the rhs along the tile's `n`, or along
    /// the whole of `K` for a col rhs bound in its buffer's order, every element read once per
    /// plane that reads it. Its drain, each plane writing its tile a
    /// row of it at a time. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &PlaneFragmentMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, partitioning) = (
            MatmulView::of::<PlaneFragmentMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &blueprint.partitioning,
        );
        let staging = MatmulStagingCount {
            readers: Self::readers(partitioning),
            ..MatmulStagingCount::new(
                partitioning,
                &partitioning.stage(),
                blueprint.microkernel.orders,
                &blueprint.shared_memory.ring,
                blueprint.microkernel.tile_extent(N),
            )
        };
        staging.count(&item, partitioning, estimate);
        blueprint
            .microkernel
            .estimate(&item, &(), grid.planes(), estimate);
    }

    /// A plane is as many units as its instruction runs on, and it walks its own `K` below the
    /// planes' level, so each stages its own copy of the stage ([`Stages::smem`]); the ring is
    /// resolved against the one plane's units that fill it.
    ///
    /// [`Stages::smem`]: cubecl::tile::Stages::smem
    ///
    /// A fragment's store intrinsic writes its whole tile through an address, so where the tiles
    /// overhang the output, or the output is a sink with no address, the fragments drain through a
    /// scratch window, each cell landing only inside. The software instruction and a manual
    /// fragment write their cells on their own and never hold one.
    ///
    /// The shared memory a cube holds charges the window wherever its fragments can bounce:
    /// whether the tiles overhang reads the exact `m`, which a tune key anchors, and what the
    /// tuner ranked on the key's problem has to fit the problem it launches on.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_plain()?;
        problem.check_unindexed()?;
        let view = MatmulView::of::<PlaneFragmentMicrokernel>(problem)?;
        let item = view.work_item();
        let microkernel = PlaneFragmentMicrokernel::best(hw, &item, &settings.microkernel)?;
        let partitioning = view.plane_partitioning(
            microkernel.tiling().walk_every(&[K]),
            &[M, N],
            PlaneOrder::new(&settings.planes.counts(), settings.cube_order),
        );
        let window = match microkernel.instruction.drain() {
            PlaneFragmentDrain::Bounce => Scratch::OneTile,
            PlaneFragmentDrain::CellByCell | PlaneFragmentDrain::InLines => Scratch::None,
        };
        let bounces =
            partitioning.overhangs_output() || problem.destination == MatmulDestination::Sink;
        let scratch = match bounces {
            true => window,
            false => Scratch::None,
        };
        let plane_width = microkernel.instruction.units(hw)?;
        let ring = settings.ring.resolve(
            hw,
            &partitioning
                .stage()
                .operands(item.elements, microkernel.orders),
            plane_width as usize,
        );
        let stage = microkernel
            .instruction
            .stage_storage(partitioning.batch_axes(), MatmulAxes::MNK);
        let planes = settings.planes.count();
        let cube = CubeResources {
            plane_width,
            shared_memory: planes
                * (ring.bytes(partitioning.stage().bytes(
                    hw,
                    problem.elements,
                    &stage,
                    microkernel.orders,
                )) + microkernel.scratch_bytes(&item, window)),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let shared_memory = PlaneFragmentMatmulRoutineSharedMemory {
            stage,
            ring,
            scratch,
        };
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint = MatmulBlueprint::new(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            shared_memory,
        );
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<PlaneFragmentMicrokernel, PlaneFragmentMatmulRoutineSharedMemory>
    for PlaneFragmentMatmulRoutine
{
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: PlaneFragmentMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let [sum, lhs, _, out] = tensors.work_item().kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles. Its shared memory is sized by the planes of the grid it launches on: each
        // plane's stages are one copy of the planes' `grid.planes()`, filled by `CUBE_DIM_X` units.
        unsafe {
            plane_fragment_matmul_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count.clone(),
                grid.launch.cube_dim,
                blueprint.widths.out,
                binding.lhs,
                binding.rhs,
                O::arg(binding.out),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.shared_memory,
                [sum, lhs, out],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::MatmulLayout;
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentInstruction;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A device offering the `16×16×16` `f16` instruction summed in `f32`.
    fn offering(hw: Hardware) -> Hardware {
        devices::offering_cmma(
            hw,
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: F32,
                m: 16,
                n: 16,
                k: 16,
            }],
        )
    }

    /// A setting settles to one launch at an anchored row count and at a row count the key
    /// anchors to it: the tuner ranks a candidate on the first and launches it on the second, so
    /// what one cube holds, its shape and its ring are the same for both, and so is a refusal.
    /// 244 rows cut a tile short that 256 do not, which is where the fragments bounce.
    #[test]
    fn a_setting_settles_alike_at_the_rows_a_key_anchors() {
        use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
        use cubek_std::matmul_axes::N;
        let hw = offering(devices::strix_halo());
        let shape = |m, n, k, rhs| MatmulShape {
            m,
            n,
            k,
            batch: 1,
            elem: F16,
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(Some(rhs), MatmulBatching::Shared),
        };
        let ranked = |problem: &MatmulRoutineProblem,
                      settings: &PlaneFragmentMatmulRoutineSettings| {
            PlaneFragmentMatmulRoutine
                .prepare(&hw, problem, settings)
                .map(|(blueprint, grid)| {
                    (
                        grid.shared_memory,
                        grid.launch.cube_dim,
                        blueprint.partitioning.cube_box(),
                        blueprint.shared_memory.ring,
                    )
                })
                .map_err(|refusal| format!("{refusal:?}"))
        };
        for (n, k, rhs) in [(4096, 1024, N), (1024, 3072, K)] {
            let anchored = MatmulRoutineProblem::from(&shape(256, n, k, rhs));
            let short = MatmulRoutineProblem::from(&shape(244, n, k, rhs));
            let default = PlaneFragmentMatmulRoutineSettings::DEFAULT;
            assert!(
                ranked(&short, &default).is_ok(),
                "the default on n {n}, k {k}"
            );
            for settings in PlaneFragmentMatmulRoutine.settings(&hw) {
                assert_eq!(
                    ranked(&anchored, &settings),
                    ranked(&short, &settings),
                    "{settings} on n {n}, k {k}"
                );
            }
        }
    }

    /// Every setting prepares on a device offering the instruction and on the software
    /// instruction, ragged shapes included; one other than the default may ask for a cube the
    /// device cannot hold, and declines saying so.
    #[test]
    fn every_setting_prepares_where_planes_run() {
        for hw in [
            offering(devices::strix_halo()),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
        ] {
            for (position, settings) in PlaneFragmentMatmulRoutine.settings(&hw).iter().enumerate()
            {
                for (m, n, k) in [(1, 128, 256), (37, 29, 19), (512, 4096, 4096)] {
                    let problem = MatmulRoutineProblem::dense(m, n, k, F16, MatmulLayout::Row);
                    let prepared = PlaneFragmentMatmulRoutine.prepare(&hw, &problem, settings);
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

    /// Each plane walks its own `K`: a stage is one plane's tile a step of `K` deep, not the
    /// cube's box, and the cube holds one copy of it for every plane, beside the window each
    /// plane's fragments can bounce through.
    #[test]
    fn every_plane_stages_its_own_tile() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let settings = PlaneFragmentMatmulRoutineSettings::DEFAULT;
        let (blueprint, grid) = PlaneFragmentMatmulRoutine
            .prepare(&hw, &problem, &settings)
            .expect("the routine runs here");
        let (cube, stage) = (
            blueprint.partitioning.cube_box(),
            blueprint.partitioning.stage(),
        );
        let microkernel = blueprint.microkernel;
        assert_eq!(stage.extent(M), microkernel.tile_extent(M));
        assert_eq!(stage.extent(N), microkernel.tile_extent(N));
        assert_eq!(stage.extent(K), microkernel.tile_extent(K));
        assert_eq!(cube.extent(M), 2 * microkernel.tile_extent(M));
        let one_copy = blueprint.partitioning.stage().bytes(
            &hw,
            problem.elements,
            &blueprint.shared_memory.stage,
            microkernel.orders,
        );
        let window = microkernel.scratch_bytes(
            &MatmulView::of::<PlaneFragmentMicrokernel>(&problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            Scratch::OneTile,
        );
        assert_eq!(grid.planes().count, settings.planes.count());
        assert_eq!(
            grid.shared_memory,
            settings.planes.count() * (one_copy + window)
        );
    }

    /// A plane's ring is filled by its own units alone, so a register prefetch is resolved
    /// against one plane's width: what each unit holds is its share of one plane's stage.
    #[test]
    fn a_ring_is_resolved_against_one_planes_units() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let settings = PlaneFragmentMatmulRoutineSettings {
            ring: StageRing::DOUBLE_THROUGH_REGISTERS,
            ..PlaneFragmentMatmulRoutineSettings::DEFAULT
        };
        let (blueprint, grid) = PlaneFragmentMatmulRoutine
            .prepare(&hw, &problem, &settings)
            .expect("the routine runs here");
        let plane_units = grid.planes().width;
        assert_eq!(
            blueprint.shared_memory.ring,
            settings.ring.resolve(
                &hw,
                &blueprint.partitioning.stage().operands(
                    MatmulView::of::<PlaneFragmentMicrokernel>(&problem)
                        .expect(MatmulView::PREPARED)
                        .work_item()
                        .elements,
                    blueprint.microkernel.orders
                ),
                plane_units
            )
        );
    }

    /// The software instruction writes its cells on their own, and an output its tiles cover
    /// exactly stores a fragment through its own intrinsic: neither holds a scratch. One its tiles
    /// overhang bounces a fragment through one.
    #[test]
    fn only_an_overhanging_fragment_output_drains_through_a_scratch() {
        let scratch = |hw: &Hardware, m, n, k| {
            let problem = MatmulRoutineProblem::dense(m, n, k, F16, MatmulLayout::Row);
            let (blueprint, _) = PlaneFragmentMatmulRoutine
                .prepare(hw, &problem, &PlaneFragmentMatmulRoutineSettings::DEFAULT)
                .expect("the default prepares here");
            (
                blueprint.microkernel.instruction,
                blueprint.shared_memory.scratch,
            )
        };
        let (instruction, software) = scratch(&devices::strix_halo(), 37, 29, 19);
        assert!(matches!(
            instruction,
            PlaneFragmentInstruction::Registers { .. }
        ));
        assert_eq!(software, Scratch::None);
        let fragment = offering(devices::strix_halo());
        assert_eq!(scratch(&fragment, 256, 512, 64).1, Scratch::None);
        assert_eq!(scratch(&fragment, 37, 29, 19).1, Scratch::OneTile);
    }
}
