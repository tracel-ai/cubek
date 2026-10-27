use super::super::blueprint::hypercube_blueprint;
use super::kernel::hypercube_last_arrival_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::MatmulPartitioning;
use crate::kernels::{
    MATMUL_PACKED_WALK_PAST_K, MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, MatmulView,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Space, StageStorage, TileArgLaunch, launch::DestinationLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::cube::MatmulCubeWrite;
use cubek_microkernel::matmul::cube::{CubeStagedDelivery, CubeStagedMicrokernel};
use cubek_microkernel::matmul::hypercube::{
    HypercubeStagedMicrokernel, HypercubeStagedMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, KB, KI, M, N};
use cubek_std::microkernel::StageRing;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every tile's steps of `K` laid end to end and shared between a fixed
/// grid of cubes, one run each: a run may start inside one tile and end inside another,
/// and every run but the last holds the same number of steps, so no cube does more than its share
/// however the tiles divide the grid.
///
/// What a product with more output tiles than the device holds at once but not a whole number of
/// times over wants: one cube a tile leaves the device idle through the last round of tiles, where
/// a grid sized to the device's workers runs no such round. The grid is the device's
/// [`workers`](Hardware::workers) times a setting's count for every product of at least that many
/// tiles, so one kernel serves them all for an element and a layout.
///
/// A cube contracts its part of every tile its run touches with the cube microkernel, then hands
/// it on ([`LastArrival`](cubecl::tile::launch::LastArrival)). A tile it holds whole goes straight
/// into the output. A part of a tile several runs share goes into a slot of the sum's element, and
/// the last of them to arrive adds the parts in run order and hands the total on. No cube waits on
/// another, and the total is the same bits from run to run. Any number of runs may share a tile,
/// so a product of fewer tiles than the grid has cubes splits each tile between several. Declines
/// where one cube's stores do not reach another within a launch, and on a device that does not
/// say how many workers it has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeLastArrivalMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalMatmulRoutineSettings {
    /// What each cube runs over its part of a tile: the hypercube microkernel the relay routine
    /// runs too, since a tile several runs share is a tile cut along `K` between them.
    pub microkernel: HypercubeStagedMicrokernelSettings,
    /// Cubes the grid holds for every worker of the device.
    pub cubes_per_worker: NonZeroUsize,
}

impl HypercubeLastArrivalMatmulRoutineSettings {
    /// The staged cube's own box, four cubes a worker. A product that wants this routine has more tiles
    /// than the device holds at once, which is the problem the staged cube's box is for, and four
    /// of its cubes are what a worker holding a thousand units runs at once, so the grid is one
    /// round.
    pub const DEFAULT: Self = Self {
        microkernel: HypercubeStagedMicrokernelSettings::CUBE_BOX,
        cubes_per_worker: NonZeroUsize::new(4).unwrap(),
    };
}

impl core::fmt::Display for HypercubeLastArrivalMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-last-arrival-matmul {} cubes_per_worker={}",
            self.microkernel, self.cubes_per_worker
        )
    }
}

/// The last-arrival routine's blueprint: how a stage its cubes' microkernel walks holds its cells, as
/// the cube routine's does.
pub type HypercubeLastArrivalMatmulRoutineBlueprint =
    MatmulBlueprint<HypercubeStagedMicrokernel, StageStorage>;

impl HypercubeLastArrivalMatmulRoutine {
    /// What it declines on a device that does not say how many workers it has, the count its grid
    /// is sized by.
    pub const UNSIZED: Refusal = Refusal::Declined {
        why: "the last-arrival routine sizes its grid by the device's workers, which this device does not report",
    };

    /// The cubes its grid holds on `hw` at `settings`, or [`UNSIZED`](Self::UNSIZED) on a device
    /// that does not say how many workers it has.
    fn cubes(
        hw: &Hardware,
        settings: &HypercubeLastArrivalMatmulRoutineSettings,
    ) -> Result<NonZeroUsize, Refusal> {
        let workers = hw
            .workers()
            .and_then(|workers| NonZeroUsize::new(workers as usize))
            .ok_or(Self::UNSIZED)?;
        Ok(workers.saturating_mul(settings.cubes_per_worker))
    }

    /// `partitioning` with one cube's box stretched to its run of the line: the tiles' steps of
    /// `K`, end to end, shared between `cubes`, the longest run as one tile's contraction.
    fn one_run(partitioning: &MatmulPartitioning, cubes: usize) -> MatmulPartitioning {
        let space = partitioning.as_tile().space();
        let (boxes_m, boxes_n) = partitioning.cubes();
        let batches: usize = partitioning
            .batch_axes()
            .iter()
            .map(|&axis| space.extent(axis))
            .product();
        let tiles = boxes_m * boxes_n * batches;
        let cube = partitioning.cube_space().space;
        let depth = partitioning.stage().extent(K);
        let line = tiles * cube.extent(K).div_ceil(depth);
        let run = line.div_ceil(cubes) * depth;
        MatmulPartitioning::new(
            partitioning.as_tile().clone(),
            Space::new(&[(M, cube.extent(M)), (N, cube.extent(N)), (K, run)]),
            partitioning.batch_axes().to_vec(),
            partitioning.rows_kind(),
        )
    }
}

impl Routine for HypercubeLastArrivalMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = HypercubeStagedMicrokernel;
    type Settings = HypercubeLastArrivalMatmulRoutineSettings;
    type Blueprint = HypercubeLastArrivalMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// A star off the default: one and two cubes a worker, a second stage in flight, and the
    /// staged cube's box two instruction steps of `K` a stage, both boxes its microkernel lists
    /// ([`HypercubeStagedMicrokernel::settings`]). Nothing where one
    /// cube's stores do not reach another within a launch, or where the device does not say how
    /// many workers it has, where every one of them declines.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if !hw.properties.features.device_memory_scope || hw.workers().is_none() {
            return Vec::new();
        }
        let lead = Self::Settings::DEFAULT;
        let counts = [1, 2].map(|cubes| Self::Settings {
            cubes_per_worker: NonZeroUsize::new(cubes).unwrap(),
            ..lead
        });
        let boxes = [
            HypercubeStagedMicrokernelSettings::CUBE_BOX.ringed(StageRing::DOUBLE),
            HypercubeStagedMicrokernelSettings::CUBE_BOX_DEEP,
        ]
        .map(|microkernel| Self::Settings {
            microkernel,
            ..lead
        });
        [lead].into_iter().chain(counts).chain(boxes).collect()
    }

    /// Nothing of its own: it only distributes the cubes, and their microkernel makes every
    /// access.
    fn vectorize(
        &self,
        microkernel: &HypercubeStagedMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        assert_eq!(
            partitioning.as_tile().fillers() as usize,
            microkernel.cube.fill.filling_planes,
            "a cube is launched over its launch tiling, beside the planes its fill sets aside"
        );
        microkernel.vectorize(&partitioning.cube_space(), vectorization);
    }

    /// Its cubes' microkernel's, over one cube's run as if it were one tile's contraction: a run
    /// reads as many steps of the operands as that, whichever tiles they lie in, and drains once
    /// for each tile it touches, which is at most one more. The launch zeroing the arrival counters
    /// is the launch's, outside the kernel's estimate.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &HypercubeLastArrivalMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let partitioning = &blueprint.partitioning;
        let cubes = grid.cubes().expect("a static grid");
        blueprint.microkernel.estimate(
            &MatmulView::of::<HypercubeStagedMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &Self::one_run(partitioning, cubes as usize).cube_space(),
            grid.planes(),
            estimate,
        );
    }

    /// A plane is as many units as its instruction runs on. The grid is the device's workers times
    /// the setting's count, and each cube resolves its microkernel merging the output by the last
    /// arrival. The slots hold the sum's element and the merge copies one into the output line for
    /// line, so the output is no wider than a line of the sum.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_unindexed()?;
        let view = MatmulView::of::<HypercubeStagedMicrokernel>(problem)?;
        let item = view.work_item();
        let microkernel = HypercubeStagedMicrokernel::best(hw, &item, &settings.microkernel)?;
        if microkernel.cube.walk().decoded().is_some() {
            view.check_steps(hw)?;
        }
        let target = Self::cubes(hw, settings)?;
        // Against packed inputs each cube contracts them as its microkernel reads them, over the
        // block-split space, its share of the walk whole blocks of `K`.
        let axes = [item.axes.m, item.axes.n];
        let resolved = microkernel.cube.resolve(
            hw,
            &microkernel.cube.contracted(&item),
            &problem.buffers(microkernel.cube.plane.orders),
            <Self as MatmulRoutine<HypercubeStagedMicrokernel, StageStorage>>::WRITE,
            &view
                .shared_partitioning(microkernel.cube.tiling(), &axes, target)
                .cube_space(),
        )?;
        let partitioning =
            view.shared_partitioning(resolved.microkernel.launch_tiling(), &axes, target);
        let past_k = || {
            [KB, KI]
                .into_iter()
                .any(|axis| partitioning.as_tile().overhangs(axis))
        };
        if item.blocks.is_some() && past_k() {
            return Err(MATMUL_PACKED_WALK_PAST_K);
        }
        let storage = resolved.storage;
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        let microkernel = HypercubeStagedMicrokernel {
            cube: resolved.microkernel,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let blueprint = hypercube_blueprint(hw, problem, &item, microkernel, partitioning, storage);
        Ok((blueprint, grid))
    }
}

impl HypercubeLastArrivalMatmulRoutine {
    /// Cubes a worker the settings against a packed input race.
    const PACKED_CUBES_PER_WORKER: [usize; 3] = [1, 2, 4];

    /// The settings a product against a packed input races on `hw`: each box the cube staged
    /// microkernel races against one ([`CubeStagedMicrokernel::packed_settings`]) whose planes
    /// each drain their own tile and set none aside to fill, which is what a hypercube's cubes run,
    /// at one, two and four cubes a worker. None where the routine's own list is empty: on a
    /// device whose cubes do not meet within a launch, or that reports no workers.
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<HypercubeLastArrivalMatmulRoutineSettings> {
        if self.settings(hw).is_empty() {
            return Vec::new();
        }
        CubeStagedMicrokernel::packed_settings(hw)
            .into_iter()
            .filter(|cube| {
                cube.planes_along_k == NonZeroUsize::MIN && cube.fill.filling_planes == 0
            })
            .flat_map(|cube| {
                Self::PACKED_CUBES_PER_WORKER.map(|cubes| {
                    HypercubeLastArrivalMatmulRoutineSettings {
                        microkernel: HypercubeStagedMicrokernelSettings { cube },
                        cubes_per_worker: NonZeroUsize::new(cubes).expect("a count of cubes"),
                    }
                })
            })
            .collect()
    }
}

impl MatmulRoutine<HypercubeStagedMicrokernel, StageStorage> for HypercubeLastArrivalMatmulRoutine {
    const WRITE: MatmulCubeWrite = MatmulCubeWrite::LastArrival;

    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: HypercubeLastArrivalMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        // The elements its cubes contract: packed inputs a fill decodes at the element they decode
        // into, ones read as stored at the element their values name.
        let elements = blueprint
            .microkernel
            .cube
            .contracted(&tensors.work_item())
            .kernel_elements();
        let stage = blueprint.partitioning.stage();
        let merge = tensors.bind_last_arrival(&blueprint, &binding.launcher);
        let out_width = blueprint.widths.out;
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, or by a descriptor, whose copies fill what lies
        // past the operand with zeros; the kernel indexes no buffer but through its tiles and the
        // arrival counter of each tile its run touches. Its shared memory is sized by the planes of
        // the grid it launches on: the stage fill counts `grid.launch.cube_dim`'s units, the
        // scratch windows `grid.planes()`. The counters arrive holding zero.
        match blueprint.microkernel.cube.fill.delivery {
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => unsafe {
                hypercube_last_arrival_matmul_kernel::launch_unchecked::<O>(
                    tensors.client(),
                    grid.launch.cube_count.clone(),
                    grid.launch.cube_dim,
                    out_width,
                    binding.lhs,
                    binding.rhs,
                    O::arg(binding.out),
                    merge.slots.arg(),
                    merge.arrivals,
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    blueprint.shared_memory,
                    elements,
                );
            },
            CubeStagedDelivery::Tma => {
                let (lhs_map, rhs_map) = tensors.tma_operands(
                    &stage,
                    &blueprint.shared_memory,
                    blueprint.microkernel.cube.plane.orders,
                );
                unsafe {
                    hypercube_last_arrival_matmul_kernel::launch_unchecked::<O>(
                        tensors.client(),
                        grid.launch.cube_count.clone(),
                        grid.launch.cube_dim,
                        out_width,
                        lhs_map,
                        rhs_map,
                        O::arg(binding.out),
                        merge.slots.arg(),
                        merge.arrivals,
                        binding.launcher.partitioning_arg(),
                        blueprint.microkernel,
                        blueprint.shared_memory,
                        elements,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::MatmulLayout;
    use cubek_microkernel::matmul::cube::CubeStagedMicrokernelSettings;
    use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn offering(hw: Hardware) -> Hardware {
        devices::offering_cmma(
            hw,
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 16,
                k: 16,
            }],
        )
    }

    /// A setting asking for the warpgroup MMA declines, on a device and product it runs on, with
    /// its stages landed by the TMA engine and planes set aside to fill them or none: the
    /// hypercube's cubes fill with every plane, and its walk releases each stage as its consume
    /// returns, before the MMAs reading it are done.
    #[test]
    fn declines_the_warpgroup_mma() {
        use cubecl::features::{WgmmaConfig, WgmmaElems};
        use cubek_microkernel::matmul::cube::{CubeStagedDelivery, CubeStagedFill};
        use cubek_microkernel::matmul::plane::PlaneFragmentTransport;
        use cubek_std::microkernel::StageRing;

        let hw = devices::offering_wgmma(
            offering(devices::h100()),
            [WgmmaConfig {
                elems: WgmmaElems {
                    a: F16,
                    b: F16,
                    cd: ElemType::Float(FloatKind::F32),
                },
                m: 64,
                n_granularity: 8,
                n_max: 256,
                k: 16,
            }],
        );
        let problem = MatmulRoutineProblem::dense(128, 256, 512, F16)
            .laid(MatmulOperand::Rhs, MatmulLayout::Col);
        let lead = HypercubeLastArrivalMatmulRoutineSettings::DEFAULT;
        let cube = lead.microkernel.cube;
        for filling_planes in [0, 1] {
            let mut settings = lead;
            settings.microkernel.cube = CubeStagedMicrokernelSettings {
                plane: PlaneFragmentMicrokernelSettings {
                    along_m: NonZeroUsize::MIN,
                    along_n: NonZeroUsize::new(16).unwrap(),
                    depth: NonZeroUsize::new(4).unwrap(),
                    transport: PlaneFragmentTransport::Wgmma,
                    ..cube.plane
                },
                ring: StageRing::DOUBLE,
                fill: CubeStagedFill {
                    delivery: CubeStagedDelivery::Tma,
                    filling_planes,
                },
                ..cube
            };
            let declined = HypercubeLastArrivalMatmulRoutine.prepare(&hw, &problem, &settings);
            assert!(
                matches!(declined, Err(Refusal::Declined { .. })),
                "{settings}: {declined:?}"
            );
        }
    }

    /// Every setting it lists prepares, on a ragged problem, on an integrated part and on an
    /// `H100`, devices where one cube's stores reach another within a launch; its cubes walk the
    /// stage the placement states.
    #[test]
    fn every_setting_prepares_where_cubes_meet() {
        let problem = MatmulRoutineProblem::dense(37, 29, 300, F16);
        for hw in [devices::strix_halo(), devices::h100()].map(offering) {
            for settings in HypercubeLastArrivalMatmulRoutine.settings(&hw) {
                let (blueprint, _) = HypercubeLastArrivalMatmulRoutine
                    .prepare(&hw, &problem, &settings)
                    .unwrap_or_else(|refusal| panic!("{settings}: {refusal:?}"));
                let partitioning = &blueprint.partitioning;
                assert_eq!(
                    blueprint.microkernel.cube.stage(&partitioning.cube_space()),
                    partitioning.stage(),
                    "{settings}"
                );
            }
        }
    }

    /// Where one cube's stores do not reach another within a launch, which is how its cubes meet,
    /// it declines with the reason its microkernel names; where the device does not say how many
    /// workers it has, with its own.
    #[test]
    fn declines_where_cubes_cannot_meet_or_the_grid_has_no_size() {
        let problem = MatmulRoutineProblem::dense(64, 64, 512, F16);
        let settings = HypercubeLastArrivalMatmulRoutineSettings::DEFAULT;
        let isolated = {
            let mut hw = offering(devices::strix_halo());
            devices::features(&mut hw).device_memory_scope = false;
            hw
        };
        let declined = HypercubeLastArrivalMatmulRoutine.prepare(&isolated, &problem, &settings);
        assert!(
            matches!(
                declined,
                Err(Refusal::Declined {
                    why: HypercubeStagedMicrokernel::DECLINE
                })
            ),
            "{declined:?}"
        );
        let uncounted = {
            let mut hw = offering(devices::strix_halo());
            devices::topology(&mut hw).num_streaming_multiprocessors = None;
            hw
        };
        let declined = HypercubeLastArrivalMatmulRoutine.prepare(&uncounted, &problem, &settings);
        assert!(
            matches!(declined, Err(HypercubeLastArrivalMatmulRoutine::UNSIZED)),
            "{declined:?}"
        );
    }

    /// The grid is the device's workers times the setting's count, whatever the product: forty
    /// workers at four a worker spawn a hundred and sixty cubes for a product of fewer tiles, of as
    /// many and of many more, and each cube's box holds the whole of `K`.
    #[test]
    fn the_grid_is_the_devices_whatever_the_product() {
        let hw = offering(devices::strix_halo());
        let settings = HypercubeLastArrivalMatmulRoutineSettings::DEFAULT;
        for (m, n, k) in [(159 * 128, 256, 512), (2560, 2048, 256), (2048, 4096, 1024)] {
            let problem = MatmulRoutineProblem::dense(m, n, k, F16);
            let (blueprint, grid) = HypercubeLastArrivalMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the routine runs here");
            assert_eq!(grid.cubes(), Some(160), "{m}x{n}x{k}");
            assert_eq!(
                blueprint.partitioning.cube_space().extent(K),
                k,
                "{m}x{n}x{k}"
            );
        }
    }

    /// A cube is estimated by its run of the line, whichever tiles it lies in: `4096 × 2048 × 1024`
    /// in the default's `128 × 256` boxes, sixteen of `K` a stage, is two hundred and fifty-six
    /// boxes of sixty-four stages, which a hundred and sixty cubes share in runs of a hundred and
    /// three stages.
    #[test]
    fn a_cube_is_estimated_by_its_run() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(4096, 2048, 1024, F16);
        let (blueprint, grid) = HypercubeLastArrivalMatmulRoutine
            .prepare(
                &hw,
                &problem,
                &HypercubeLastArrivalMatmulRoutineSettings::DEFAULT,
            )
            .expect("the routine runs here");
        let partitioning = &blueprint.partitioning;
        let cube = partitioning.cube_space().space;
        assert_eq!((cube.extent(M), cube.extent(N)), (128, 256));
        assert_eq!(partitioning.stage().extent(K), 16);
        let run = HypercubeLastArrivalMatmulRoutine::one_run(partitioning, 160)
            .cube_space()
            .space;
        assert_eq!(grid.cubes(), Some(160));
        assert_eq!(
            (run.extent(M), run.extent(N), run.extent(K)),
            (128, 256, 103 * 16)
        );
    }
}
