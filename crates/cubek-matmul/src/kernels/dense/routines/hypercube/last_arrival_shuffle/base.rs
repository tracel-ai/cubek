use super::kernel::hypercube_last_arrival_shuffle_matmul_kernel;
use crate::kernels::{
    HypercubeLastArrivalMatmulRoutine, MATMUL_HYPERCUBE_PACKED_LHS, MATMUL_PACKED_WALK_PAST_K,
    MATMUL_STREAMED_INPUT_FLOAT, MatmulBinding, MatmulBlueprint, MatmulOutput, MatmulRoutine,
    MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use core::num::NonZeroUsize;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::cube::MatmulCubeWrite;
use cubek_microkernel::matmul::plane::{
    PlaneShuffleFetch, PlaneShuffleMicrokernel, PlaneShuffleMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, KB, KI, M, blocked};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` against a packed rhs, with every box's steps of `K` laid end to end and
/// shared between a fixed grid of cubes, one run each, as [`HypercubeLastArrivalMatmulRoutine`]
/// shares them, but each cube one plane of the shuffle ([`PlaneShuffleMicrokernel`]) summing its
/// run in its units' registers rather than staging it for a matrix instruction: what a product of
/// few rows, bound by the bytes of its rhs, wants. Its units each own their columns of the box and
/// walk the run's steps, so a box's partials meet once, in its slots, rather than at every cube.
///
/// The grid is the device's [`workers`](Hardware::workers) times a setting's count, so every
/// cube's run is the same share of the line whatever the box count, and declines where one
/// cube's stores do not reach another within a launch, on a device that does not say how many
/// workers it has, against a float rhs or a packed lhs, and on a block whose units walk runs of
/// `K` of their own, which the cubes cannot share.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeLastArrivalShuffleMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalShuffleMatmulRoutineSettings {
    /// What each cube's plane runs over its part of a box.
    pub microkernel: PlaneShuffleMicrokernelSettings,
    /// Cubes the grid holds for every worker of the device.
    pub cubes_per_worker: NonZeroUsize,
}

impl core::fmt::Display for HypercubeLastArrivalShuffleMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-last-arrival-shuffle-matmul {} cubes_per_worker={}",
            self.microkernel, self.cubes_per_worker
        )
    }
}

pub type HypercubeLastArrivalShuffleMatmulRoutineBlueprint =
    MatmulBlueprint<PlaneShuffleMicrokernel>;

impl HypercubeLastArrivalShuffleMatmulRoutine {
    /// What a block declines whose units each walk a run of `K` of their own, below the plane's
    /// steps: the cubes share a box's steps, which such a block has none of above its units.
    pub const RUNS_BELOW_THE_UNITS: Refusal = Refusal::Declined {
        why: "the shuffle hypercube shares a box's steps of K between cubes, and this block's units each walk a run of their own",
    };

    /// Cubes a worker the settings race: a cube is one plane, so a worker holds many.
    const CUBES_PER_WORKER: [usize; 3] = [4, 8, 16];

    /// The settings worth racing against a packed rhs on `hw`: each count of steps of `N` a unit
    /// and of rows a unit the plane routine races against one
    /// ([`PlaneShuffleMatmulRoutine::packed_settings`](crate::kernels::PlaneShuffleMatmulRoutine::packed_settings)),
    /// at each count of cubes a worker, its steps' loads issued in the step and a step ahead.
    /// None on a device whose cubes do not meet within a launch, or that reports no workers.
    pub fn packed_settings(
        &self,
        hw: &Hardware,
    ) -> Vec<HypercubeLastArrivalShuffleMatmulRoutineSettings> {
        if !hw.properties.features.device_memory_scope || hw.workers().is_none() {
            return Vec::new();
        }
        use PlaneShuffleMicrokernelSettings as Block;
        let blocks = [
            Block::stepping(1, None),
            Block::stepping(2, None),
            Block::stepping(2, Some(2)),
            Block::stepping(4, Some(2)),
        ];
        let fetches = [PlaneShuffleFetch::InStep, PlaneShuffleFetch::StepAhead];
        fetches
            .into_iter()
            .flat_map(|fetch| {
                blocks.into_iter().flat_map(move |block| {
                    Self::CUBES_PER_WORKER.map(|cubes| {
                        HypercubeLastArrivalShuffleMatmulRoutineSettings {
                            microkernel: block.fetching(fetch),
                            cubes_per_worker: NonZeroUsize::new(cubes).expect("a count of cubes"),
                        }
                    })
                })
            })
            .collect()
    }
}

impl Routine for HypercubeLastArrivalShuffleMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = PlaneShuffleMicrokernel;
    type Settings = HypercubeLastArrivalShuffleMatmulRoutineSettings;
    type Blueprint = HypercubeLastArrivalShuffleMatmulRoutineBlueprint;
    type Partitioning = crate::kernels::MatmulPartitioning;

    /// Its packed settings: it reads only a packed rhs.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        self.packed_settings(hw)
    }

    /// Nothing of its own: it only distributes the cubes, and their plane makes every access.
    fn vectorize(
        &self,
        microkernel: &PlaneShuffleMicrokernel,
        _: &crate::kernels::MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Every byte of the rhs once, its words and scales, and the lhs's rows once for each run of
    /// a box's columns, over the whole grid; the slots a run parks and the last arrival reads are
    /// the sum's element, a box of rows and columns, at most twice a cube. Then its plane, for the
    /// work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &HypercubeLastArrivalShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let view = MatmulView::of::<PlaneShuffleMicrokernel>(problem).expect(MatmulView::PREPARED);
        let (item, microkernel) = (view.work_item(), &blueprint.microkernel);
        let packing = problem
            .packings
            .rhs
            .expect("the shuffle hypercube prepares only against a packed rhs");
        let [rows, k, n] = [view.rows(), problem.k, problem.n].map(|extent| extent as u64);
        let leaf = blueprint.partitioning.leaf();
        let (blocks, inside) = blocked(K);
        let step_k = (leaf.extent(blocks) * leaf.extent(inside) * microkernel.units_along_k) as u64;
        let box_columns = (microkernel.columns() * microkernel.groups_along_n) as u64;
        let size = |operand: MatmulOperand| item.elements.get(operand).size() as u64;
        let rhs = packing.bytes(k as usize * n as usize) as u64;
        let cubes = grid.cubes().expect("a static grid");
        let slots = 2 * cubes * rows * box_columns * item.sum.size() as u64;
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * n * k) as f64,
            reads: vec![
                GlobalStream {
                    bytes: (n / box_columns.max(1)) * rows * k * size(MatmulOperand::Lhs),
                    run_bytes: (step_k * size(MatmulOperand::Lhs)) as usize,
                },
                GlobalStream {
                    bytes: rhs,
                    run_bytes: (rhs / n.max(1)) as usize,
                },
                GlobalStream {
                    bytes: slots,
                    run_bytes: (box_columns * item.sum.size() as u64) as usize,
                },
            ],
            writes: vec![
                GlobalStream {
                    bytes: rows * n * size(MatmulOperand::Out),
                    run_bytes: (box_columns * size(MatmulOperand::Out)) as usize,
                },
                GlobalStream {
                    bytes: slots,
                    run_bytes: (box_columns * item.sum.size() as u64) as usize,
                },
            ],
            stages: k.div_ceil(step_k.max(1)),
            stages_in_flight: 1,
            drain_atomics: 0,
        });
        microkernel.estimate(&item, &(), grid.planes(), estimate);
    }

    /// The plane the shuffle runs on a packed rhs at the settings' block, its steps of `K` above
    /// its units shared between the device's workers times the settings' count of cubes, each
    /// cube one plane over a box of the columns its units own.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_unindexed()?;
        settings.microkernel.fetch.admits(&problem.packings)?;
        let packing = problem.packings.rhs.ok_or(MATMUL_STREAMED_INPUT_FLOAT)?;
        if problem.packings.lhs.is_some() {
            return Err(MATMUL_HYPERCUBE_PACKED_LHS);
        }
        let view = MatmulView::of::<PlaneShuffleMicrokernel>(problem)?;
        view.check_steps(hw)?;
        let microkernel =
            PlaneShuffleMicrokernel::best(hw, &view.work_item(), &settings.microkernel)?;
        let Some(tiling) = microkernel.tiling_over_rows() else {
            return Err(Self::RUNS_BELOW_THE_UNITS);
        };
        let target = HypercubeLastArrivalMatmulRoutine::cubes(hw, settings.cubes_per_worker)?;
        // The cubes cut the rows too, one box of the plane's rows, which its tiling states: the
        // slots a box parks its parts in are laid out by the axes the cubes cut.
        let (columns, _) = microkernel.column_cut();
        let partitioning = view.shared_partitioning(tiling, &[blocked(M).1, columns], target);
        let past_k = [KB, KI]
            .into_iter()
            .any(|axis| partitioning.as_tile().overhangs(axis));
        if past_k {
            return Err(MATMUL_PACKED_WALK_PAST_K);
        }
        let cube = CubeResources {
            plane_width: microkernel.plane_width() as u32,
            shared_memory: 0,
        };
        let blueprint = MatmulBlueprint::packed(&packing, microkernel, partitioning);
        let grid = Grid::new(hw, blueprint.partitioning.as_tile(), cube)?;
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<PlaneShuffleMicrokernel> for HypercubeLastArrivalShuffleMatmulRoutine {
    const WRITE: MatmulCubeWrite = MatmulCubeWrite::LastArrival;

    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileTensorLaunch, TileTensorLaunch>,
        blueprint: HypercubeLastArrivalShuffleMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let [sum, _, _, out] = tensors.work_item().kernel_elements();
        let merge = tensors.bind_last_arrival(&blueprint, &binding.launcher);
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a box the problem's edge cuts short; the kernel indexes no buffer but through its
        // tiles and the arrival counter of each box its run touches. It holds no shared memory but
        // the arrival flag the merge declares. The counters arrive holding zero.
        unsafe {
            hypercube_last_arrival_shuffle_matmul_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count.clone(),
                grid.launch.cube_dim,
                blueprint.widths.out,
                binding.lhs,
                binding.rhs,
                O::arg(binding.out),
                merge.slots.arg(),
                merge.arrivals,
                binding.launcher.partitioning_arg(),
                blueprint.microkernel.tile_instruction(),
                blueprint.microkernel.settings.fetch,
                [sum, out],
            );
        }
    }
}
