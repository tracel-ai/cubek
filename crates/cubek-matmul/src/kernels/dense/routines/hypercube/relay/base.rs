use super::kernel::hypercube_relay_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{CUBE_STRIPS, cube_order_name};
use crate::kernels::{
    MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::VectorRequirement;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::CubeOrder;
use cubecl::tile::{Scratch, StageStorage, TileArgLaunch, launch::DestinationLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulPartitioning;
use cubek_microkernel::matmul::dense::cube::CubeStagedDelivery;
use cubek_microkernel::matmul::dense::hypercube::{
    HypercubeStagedMicrokernel, HypercubeStagedMicrokernelSettings,
};
use cubek_microkernel::matmul::staged_walk::MatmulCubeWrite;
use cubek_microkernel::staging::StageRing;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with several cubes owning every tile of the output, each contracting its own
/// run of `K`, then taking its turn at the tile.
///
/// What a product with too few output tiles to fill the device wants when its contraction is long:
/// the cubes a tile would leave idle take a share of its `K` instead, and no operand is read
/// twice. Each cube runs the cube microkernel over its run, its planes laid out along `m` and `n`
/// sharing its stage; a box's cubes ride the grid's third dimension beside the batches.
///
/// The cubes of a box meet in turns, in the order of their runs ([`Relay`](cubecl::tile::launch::Relay)):
/// the first writes its sum into a carry of the sum's element, each after adds its own into it a
/// line at a time, and the last hands the total on to the output. The order is fixed, so its sums
/// are the same bits from run to run. Declines where one cube's stores do not reach another within
/// a launch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeRelayMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeRelayMatmulRoutineSettings {
    pub microkernel: HypercubeStagedMicrokernelSettings,
    /// Cubes sharing one box, each on its own run of `K`: part of the partitioning, which
    /// distributes it whatever the walk ([`MatmulPartitioning::new`]).
    pub cubes_along_k: NonZeroUsize,
    pub cube_order: CubeOrder,
}

impl HypercubeRelayMatmulRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: HypercubeStagedMicrokernelSettings::DEFAULT,
        cubes_along_k: NonZeroUsize::new(4).unwrap(),
        cube_order: CubeOrder::RowMajor,
    };
}

impl core::fmt::Display for HypercubeRelayMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-relay-matmul {} cubes_along_k={}{}",
            self.microkernel,
            self.cubes_along_k,
            cube_order_name(self.cube_order)
        )
    }
}

/// The hypercube routine's blueprint: how a stage its cubes' microkernel walks holds its cells,
/// as the cube routine's does.
pub type HypercubeRelayMatmulRoutineBlueprint =
    MatmulBlueprint<HypercubeStagedMicrokernel, StageStorage>;

impl Routine for HypercubeRelayMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = HypercubeStagedMicrokernel;
    type Settings = HypercubeRelayMatmulRoutineSettings;
    type Blueprint = HypercubeRelayMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// A star off the default: two, eight and sixteen cubes on a box, the staged cube's box, the
    /// whole partition resident in the scratch, a second stage in flight, filled in its slot and
    /// through registers, and the cubes in strips. Then the square and deep-along-`m` boxes of four
    /// planes at two, three and four cubes on a box: how many cubes share a box is what decides
    /// whether that box's grid fills the device, so each box is listed at every count. Then the
    /// default's grid two and four fragments deep on two by two and two by one planes, at two and
    /// four cubes on a box: a col weight runs along `K`, which a stage one fragment deep reads in
    /// short runs. Then the four deep grid on two by two planes, and the deep box along `m` at four
    /// cubes, with their stages' rows in order. Every box is one its microkernel lists
    /// ([`HypercubeStagedMicrokernel::settings`]). Nothing where one cube's stores do not reach
    /// another within a launch, where every one of them declines.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        use HypercubeStagedMicrokernelSettings as Shared;
        if !hw.properties.features.device_memory_scope {
            return Vec::new();
        }
        let lead = Self::Settings::DEFAULT;
        let on = |microkernel, cubes| Self::Settings {
            microkernel,
            cubes_along_k: NonZeroUsize::new(cubes).unwrap(),
            ..lead
        };
        let default = Shared::DEFAULT;
        let star = [
            on(default, 2),
            on(default, 8),
            on(default, 16),
            on(Shared::CUBE_BOX, 4),
            on(default.with_scratch(Scratch::WholeGrid), 4),
            on(default.ringed(StageRing::DOUBLE), 4),
            on(default.ringed(StageRing::DOUBLE_THROUGH_REGISTERS), 4),
        ];
        let cube_orders = CUBE_STRIPS.map(|cube_order| Self::Settings { cube_order, ..lead });
        let split_boxes = [Shared::SQUARE, Shared::DEEP_ALONG_M]
            .into_iter()
            .flat_map(move |shared| [2, 3, 4].map(move |cubes| on(shared, cubes)));
        let deep_boxes = [2, 4].into_iter().flat_map(move |depth| {
            [(2, 2), (2, 1)].into_iter().flat_map(move |planes| {
                [2, 4].map(move |cubes| on(Shared::deep(depth, planes), cubes))
            })
        });
        let rows_in_order = [
            on(Shared::deep(4, (2, 2)).rows_in_order(), 2),
            on(Shared::deep(4, (2, 2)).rows_in_order(), 4),
            on(Shared::DEEP_ALONG_M.rows_in_order(), 4),
        ];
        [lead]
            .into_iter()
            .chain(star)
            .chain(cube_orders)
            .chain(split_boxes)
            .chain(deep_boxes)
            .chain(rows_in_order)
            .collect()
    }

    /// Nothing of its own: it only distributes the cubes, and their microkernel makes every
    /// access.
    fn vectorize(
        &self,
        microkernel: &HypercubeStagedMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Nothing of its own: its cubes' microkernel walks each cube's run of `K`, and drains into
    /// the carry its turn reads back. The launch zeroing the turn counters is the launch's, outside
    /// the kernel's estimate.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &HypercubeRelayMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint.microkernel.estimate(
            &MatmulView::of::<HypercubeStagedMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &blueprint.partitioning,
            grid.planes(),
            estimate,
        );
    }

    /// A plane is as many units as its instruction runs on. The cubes sharing a box are its
    /// settings' count, whatever the walk ([`MatmulPartitioning::new`]), and each resolves its
    /// microkernel relaying the output. The carry holds the sum's element and the last turn copies
    /// it into the output line for line, so the output is no wider than a line of the sum.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_plain()?;
        problem.check_unindexed()?;
        let view = MatmulView::of::<HypercubeStagedMicrokernel>(problem)?;
        let item = view.work_item();
        let microkernel = HypercubeStagedMicrokernel::best(hw, &item, &settings.microkernel)?;
        let place = |tiling| {
            view.partitioning(tiling, &[M, N], settings.cube_order, settings.cubes_along_k)
        };
        let resolved = microkernel.cube.resolve(
            hw,
            &item,
            &problem.buffers(microkernel.cube.plane.orders),
            <Self as MatmulRoutine<HypercubeStagedMicrokernel, StageStorage>>::WRITE,
            &place(microkernel.cube.tiling()),
        )?;
        let (partitioning, storage) = (
            place(resolved.microkernel.launch_tiling()),
            resolved.storage,
        );
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        let microkernel = HypercubeStagedMicrokernel {
            cube: resolved.microkernel,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        vectorization.require(VectorRequirement::Access {
            operand: MatmulOperand::Out,
            elements: hw.vector_ceiling(MatmulView::SUM),
        });
        let mut blueprint = MatmulBlueprint::new(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            storage,
        );
        blueprint.microkernel.cube.fill = blueprint
            .microkernel
            .cube
            .fill
            .at_widths(&blueprint.widths, &item);
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<HypercubeStagedMicrokernel, StageStorage> for HypercubeRelayMatmulRoutine {
    const WRITE: MatmulCubeWrite = MatmulCubeWrite::Relay;

    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: HypercubeRelayMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let item = tensors.work_item();
        let stage = blueprint.partitioning.stage();
        let relay = tensors.bind_relay(&blueprint, &binding.launcher);
        let out_width = blueprint.widths.out;
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, or by a descriptor, whose copies fill what lies
        // past the operand with zeros; the kernel indexes no buffer but through its tiles and its
        // box's turn counter. Its shared memory is sized by the planes of the grid it launches on:
        // the stage fill counts `grid.launch.cube_dim`'s units, the scratch windows
        // `grid.planes()`. The counters arrive holding zero.
        match blueprint.microkernel.cube.fill.delivery {
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => unsafe {
                hypercube_relay_matmul_kernel::launch_unchecked::<O>(
                    tensors.client(),
                    grid.launch.cube_count.clone(),
                    grid.launch.cube_dim,
                    out_width,
                    binding.lhs,
                    binding.rhs,
                    O::arg(binding.out),
                    relay.carry.arg(),
                    relay.turns,
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    blueprint.shared_memory,
                    item.kernel_elements(),
                );
            },
            CubeStagedDelivery::Tma => {
                let (lhs_map, rhs_map) = tensors.tma_operands(
                    &stage,
                    &blueprint.shared_memory,
                    blueprint.microkernel.cube.plane.orders,
                );
                unsafe {
                    hypercube_relay_matmul_kernel::launch_unchecked::<O>(
                        tensors.client(),
                        grid.launch.cube_count.clone(),
                        grid.launch.cube_dim,
                        out_width,
                        lhs_map,
                        rhs_map,
                        O::arg(binding.out),
                        relay.carry.arg(),
                        relay.turns,
                        binding.launcher.partitioning_arg(),
                        blueprint.microkernel,
                        blueprint.shared_memory,
                        item.kernel_elements(),
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
    use cubecl::tile::space::CubeOrder;
    use cubek_microkernel::matmul::MatmulLayout;
    use cubek_microkernel::matmul::dense::cube::CubeStagedMicrokernelSettings;
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentMicrokernelSettings;
    use cubek_std::devices;
    use cubek_std::matmul_axes::K;
    use cubek_std::microkernel::GlobalStream;

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
        use cubek_microkernel::matmul::dense::cube::{CubeStagedDelivery, CubeStagedFill};
        use cubek_microkernel::matmul::dense::plane::PlaneFragmentTransport;
        use cubek_microkernel::staging::StageRing;

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
        let problem = MatmulRoutineProblem::dense(128, 256, 512, F16, MatmulLayout::Col);
        let lead = HypercubeRelayMatmulRoutineSettings::DEFAULT;
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
            let declined = HypercubeRelayMatmulRoutine.prepare(&hw, &problem, &settings);
            assert!(
                matches!(declined, Err(Refusal::Declined { .. })),
                "{settings}: {declined:?}"
            );
        }
    }

    /// Every setting it lists prepares, on a ragged problem, on a device where one cube's stores
    /// reach another within a launch.
    #[test]
    fn every_setting_prepares_where_cubes_meet() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(37, 29, 300, F16, MatmulLayout::Row);
        for settings in HypercubeRelayMatmulRoutine.settings(&hw) {
            let prepared = HypercubeRelayMatmulRoutine.prepare(&hw, &problem, &settings);
            assert!(prepared.is_ok(), "{settings}: {prepared:?}");
        }
    }

    /// Where one cube's stores do not reach another within a launch, which is how its cubes meet,
    /// it declines with the reason its microkernel names.
    #[test]
    fn declines_where_cubes_cannot_meet() {
        let hw = {
            let mut hw = offering(devices::strix_halo());
            devices::features(&mut hw).device_memory_scope = false;
            hw
        };
        let problem = MatmulRoutineProblem::dense(64, 64, 512, F16, MatmulLayout::Row);
        let declined = HypercubeRelayMatmulRoutine.prepare(
            &hw,
            &problem,
            &HypercubeRelayMatmulRoutineSettings::DEFAULT,
        );
        assert!(
            matches!(
                declined,
                Err(Refusal::Declined {
                    why: HypercubeStagedMicrokernel::DECLINE
                })
            ),
            "{declined:?}"
        );
    }

    /// Every setting it lists prepares on an `H100` too, whose shared memory and instructions are
    /// not the integrated part's the first test prepares on.
    #[test]
    fn every_setting_prepares_on_an_h100() {
        let hw = offering(devices::h100());
        let problem = MatmulRoutineProblem::dense(37, 29, 300, F16, MatmulLayout::Row);
        for settings in HypercubeRelayMatmulRoutine.settings(&hw) {
            let prepared = HypercubeRelayMatmulRoutine.prepare(&hw, &problem, &settings);
            assert!(prepared.is_ok(), "{settings}: {prepared:?}");
        }
    }

    /// The default's cube box is `64 × 128`, walked sixteen of `K` a stage, four cubes on a box:
    /// `512 × 256 × 1024` spawns eight boxes' worth of cubes four times over, each covering a
    /// quarter of `K`. The count is the one asked for whatever the walk: a walk of two steps over
    /// four cubes leaves two with nothing to add, and eleven steps over eight cubes runs two a
    /// cube.
    #[test]
    fn the_cubes_share_a_box_along_k() {
        let hw = offering(devices::strix_halo());
        let settings = HypercubeRelayMatmulRoutineSettings::DEFAULT;
        let prepared = |k| {
            let problem = MatmulRoutineProblem::dense(512, 256, k, F16, MatmulLayout::Row);
            HypercubeRelayMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the routine runs here")
        };
        let (blueprint, grid) = prepared(1024);
        assert_eq!(blueprint.partitioning.cube_box().extent(K), 256);
        assert_eq!(grid.cubes(), Some(8 * 2 * 4));
        let (blueprint, grid) = prepared(32);
        assert_eq!(grid.cubes(), Some(8 * 2 * 4));
        assert_eq!(blueprint.partitioning.cube_order(), CubeOrder::RowMajor);
        let eight = HypercubeRelayMatmulRoutineSettings {
            cubes_along_k: NonZeroUsize::new(8).unwrap(),
            ..settings
        };
        let problem = MatmulRoutineProblem::dense(512, 256, 176, F16, MatmulLayout::Row);
        let (blueprint, _) = HypercubeRelayMatmulRoutine
            .prepare(&hw, &problem, &eight)
            .expect("the routine runs here");
        assert_eq!(blueprint.partitioning.cube_box().extent(K), 32);
    }

    /// A fragment cannot land its cells in the relay's carry through its own store, so a setting
    /// asking for no scratch runs with one fragment's window.
    #[test]
    fn a_fragment_bounces_into_the_carry() {
        let hw = offering(devices::strix_halo());
        let lead = HypercubeRelayMatmulRoutineSettings::DEFAULT;
        let settings = HypercubeRelayMatmulRoutineSettings {
            microkernel: HypercubeStagedMicrokernelSettings {
                cube: CubeStagedMicrokernelSettings {
                    scratch: Scratch::None,
                    ..lead.microkernel.cube
                },
            },
            ..lead
        };
        let problem = MatmulRoutineProblem::dense(256, 512, 256, F16, MatmulLayout::Row);
        let (blueprint, _) = HypercubeRelayMatmulRoutine
            .prepare(&hw, &problem, &settings)
            .expect("the routine runs here");
        assert_eq!(blueprint.microkernel.cube.scratch, Scratch::OneTile);
    }

    /// Each cube writes its box of the output once in the operands' element, and reads and writes
    /// back its box of the carry in the sum's, which is the traffic its turn adds.
    #[test]
    fn a_turn_moves_its_box_of_the_carry_in_the_sums_element() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(256, 512, 2048, F16, MatmulLayout::Row);
        let (blueprint, grid) = HypercubeRelayMatmulRoutine
            .prepare(&hw, &problem, &HypercubeRelayMatmulRoutineSettings::DEFAULT)
            .expect("the routine runs here");
        let mut estimate = KernelEstimate::from(&grid);
        HypercubeRelayMatmulRoutine.estimate(&problem, &blueprint, &grid, &mut estimate);
        let staging = estimate.staging.expect("a staged estimate");
        let cube = blueprint.partitioning.cube_box();
        let cells = (cube.extent(M) * cube.extent(N)) as u64;
        let bytes = |streams: &[GlobalStream]| streams.iter().map(|s| s.bytes).collect::<Vec<_>>();
        assert_eq!(bytes(&staging.writes), [cells * 2, cells * 4]);
        assert_eq!(bytes(&staging.reads)[2..], [cells * 4]);
    }
}
