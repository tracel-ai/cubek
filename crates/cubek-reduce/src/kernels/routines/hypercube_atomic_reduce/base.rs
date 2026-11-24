use super::kernel::hypercube_atomic_reduce_kernel;
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem, ReduceTensors,
    each_microkernel_with, estimated_work, split_launch,
};
use core::convert::identity;
use cubecl::client::Client;
use cubecl::ir::Type;
use cubecl::ir::features::AtomicUsage;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{
    AccumulateArgLaunch, Bound, Decline, StoredTileOperand, TileOperand, Unbound, Vectorization,
};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubek_microkernel::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand, ReduceOperation};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row and adds its total into the output: a sum of few rows, too long for one
/// cube each to keep the device busy, as a whole tensor summed into one value.
///
/// The output must hold zero before it runs ([`UnitDirectZeroReduceRoutine`](crate::kernels::UnitDirectZeroReduceRoutine)
/// zeroes it), since it only ever adds.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HypercubeAtomicReduceRoutine;

/// What a measurement settles for the atomic hypercube: each cube's microkernel, and the lines a
/// unit reads of a part, which set how many cubes add into one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAtomicReduceSettings {
    pub microkernel: CubeStagedReduceMicrokernelSettings,
    pub unit_line_count: usize,
}

impl HypercubeAtomicReduceSettings {
    /// Lines each unit reads of a part, the longest parts first: the fewest atomic adds that still
    /// spread a row over the device.
    pub const UNIT_LINE_COUNTS: [usize; 3] = [32, 8, 2];
}

impl core::fmt::Display for HypercubeAtomicReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} unit_lines={}",
            self.microkernel, self.unit_line_count
        )
    }
}

pub type HypercubeAtomicReduceBlueprint = ReduceBlueprint<CubeStagedReduceMicrokernel>;

impl Routine for HypercubeAtomicReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = CubeStagedReduceMicrokernel;
    type Settings = HypercubeAtomicReduceSettings;
    type Blueprint = HypercubeAtomicReduceBlueprint;
    type Partitioning = ();

    /// Each cube microkernel setting crossed with
    /// [`UNIT_LINE_COUNTS`](HypercubeAtomicReduceSettings::UNIT_LINE_COUNTS).
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        each_microkernel_with(
            CubeStagedReduceMicrokernel::settings(hw),
            HypercubeAtomicReduceSettings::UNIT_LINE_COUNTS,
            |microkernel, unit_line_count| HypercubeAtomicReduceSettings {
                microkernel,
                unit_line_count,
            },
        )
    }

    fn vectorize(
        microkernel: &CubeStagedReduceMicrokernel,
        _partitioning: &(),
        vectorization: &mut Vectorization<ReduceOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        if problem.operation != ReduceOperation::Sum {
            return Err(Decline::Unserved {
                why: "the atomic hypercube adds its parts, which only a sum's parts make whole",
            });
        }
        if !hw
            .properties
            .atomic_type_usage(Type::atomic(problem.elements.out))
            .contains(AtomicUsage::Add)
        {
            return Err(Decline::Unserved {
                why: "the device has no atomic add for the output's element",
            });
        }
        let microkernel =
            CubeStagedReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let lane_counts = Self::lane_counts(hw, problem, &microkernel, microkernel.widest_line())?;
        let cube = CubeResources {
            plane_width: microkernel.units() as u32,
            shared_memory: microkernel.shared_memory_bytes_at_width(lane_counts[0]),
        };
        let (partitioning, grid) = split_launch(
            hw,
            problem,
            |_| problem.space(),
            cube,
            settings.unit_line_count,
            lane_counts[0],
        )?;
        Ok((
            ReduceBlueprint::new(microkernel, partitioning, lane_counts),
            grid,
        ))
    }

    fn estimate(
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, planes) = estimated_work(problem, grid);
        blueprint.microkernel.estimate(&item, &(), planes, estimate);
    }
}

impl ReduceRoutine<CubeStagedReduceMicrokernel> for HypercubeAtomicReduceRoutine {
    const NAME: &'static str = "hypercube-atomic-reduce";

    /// Never reached: [`launch`](Self::launch) and [`launch_unbound`](Self::launch_unbound) bind
    /// the output as the atomic argument the kernel adds into, which a plain output tensor is not.
    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        _problem: &ReduceRoutineProblem,
        _client: &Client,
        _binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        _blueprint: HypercubeAtomicReduceBlueprint,
        _grid: Grid,
    ) {
        unreachable!("the atomic hypercube binds its output in `launch` and `launch_unbound`")
    }

    /// `tensors` reduced on `grid`, each cube's total added into the output.
    fn launch(tensors: &ReduceTensors, blueprint: HypercubeAtomicReduceBlueprint, grid: Grid) {
        let binding = tensors.bind(&blueprint, &grid).map(Bound::arg, identity);
        launch_atomic::<StoredTileOperand>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
    }

    /// `tensors`, whose input the caller binds, reduced on `grid`, each cube's total added into
    /// the output.
    fn launch_unbound<I: TileOperand>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: HypercubeAtomicReduceBlueprint,
        grid: Grid,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) {
        let binding = tensors.bind(&blueprint, &grid).map(input, identity);
        launch_atomic::<I>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
    }
}

/// The kernel `blueprint` compiles for `problem`, launched on `grid`, its input served as `I`
/// serves it and its output bound as the atomic argument the kernel adds into.
fn launch_atomic<I: TileOperand>(
    problem: &ReduceRoutineProblem,
    client: &Client,
    binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, Bound>,
    blueprint: HypercubeAtomicReduceBlueprint,
    grid: Grid,
) {
    let elements = problem.elements;
    let (spec, lane_count) = (binding.out.spec.clone(), binding.out.vector_size);
    let out = AccumulateArgLaunch::new(binding.out.tensor(), lane_count, spec);
    // SAFETY: both operands are bound through the launcher, which masks the reads and writes of a
    // tile the problem's edge cuts short, and the kernel indexes no buffer but through its tiles;
    // the output only receives atomic adds. Its shared memory is the slots its microkernel
    // declares, which the grid counted.
    unsafe {
        hypercube_atomic_reduce_kernel::launch_unchecked::<I>(
            client,
            grid.launch.cube_count,
            grid.launch.cube_dim,
            blueprint.lane_count,
            binding.input,
            out,
            binding.launcher.partitioning_arg(),
            blueprint.microkernel,
            [elements.input, elements.accumulator, elements.out],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::{
        Reduction, bound_back, reduces_like_the_cpu, reduces_unbound_like_the_cpu, unbound_tensors,
    };
    use crate::kernels::{ReduceKernel, launch_kernel, launch_zeroed_atomic_unbound};
    use cubecl::prelude::TensorBinding;
    use cubecl::tile::launch::Decline;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// The device has no atomics.
    #[test]
    fn declines_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = HypercubeAtomicReduceRoutine::settings(&hw)[0];
        let prepared = HypercubeAtomicReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
    }

    /// An M2 Pro adds floats atomically, so it is the max that declines.
    #[test]
    fn declines_a_max() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Max);
        let settings = HypercubeAtomicReduceRoutine::settings(&hw)[0];
        let prepared = HypercubeAtomicReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
    }

    #[test]
    fn each_microkernel_setting_reads_its_longest_parts_first() {
        let hw = devices::m2_pro();
        let expected: Vec<HypercubeAtomicReduceSettings> =
            CubeStagedReduceMicrokernel::settings(&hw)
                .into_iter()
                .flat_map(|microkernel| {
                    HypercubeAtomicReduceSettings::UNIT_LINE_COUNTS.map(|unit_line_count| {
                        HypercubeAtomicReduceSettings {
                            microkernel,
                            unit_line_count,
                        }
                    })
                })
                .collect();
        assert_eq!(HypercubeAtomicReduceRoutine::settings(&hw), expected);
    }

    /// A sum of `shape` along `axis`, the only operation it serves.
    fn sum(shape: &[usize], axis: usize) -> Reduction {
        Reduction {
            shape: shape.to_vec(),
            axis,
            operation: ReduceOperation::Sum,
        }
    }

    /// The atomic hypercube launched on `tensors` as a strategy that pins it launches it, into
    /// their output, zeroed first.
    fn launch_pinned(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Decline> {
        launch_kernel(ReduceKernel::HypercubeAtomic, hw, tensors)
    }

    /// [`launch_unbound`] into `out` zeroed first by the zero routine, as its tuned candidate does.
    fn launch_zeroed_unbound(
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
        reduction: &Reduction,
    ) -> Result<(), Decline> {
        let tensors =
            unbound_tensors::<HypercubeAtomicReduceRoutine, _>(client, &input, out, reduction)?;
        let hw = Hardware::from_client(client);
        let prepared = HypercubeAtomicReduceRoutine::prepare_first_served(&hw, tensors.problem())?;
        launch_zeroed_atomic_unbound::<StoredTileOperand>(
            &hw,
            &tensors,
            prepared,
            bound_back(input),
        )
    }

    #[test]
    fn sums_one_long_row() {
        reduces_like_the_cpu(launch_pinned, sum(&[1, 1, 9000], 2));
    }

    #[test]
    fn sums_one_long_row_unbound() {
        reduces_unbound_like_the_cpu(launch_zeroed_unbound, sum(&[1, 1, 9000], 2));
    }

    #[test]
    fn sums_rows_along_the_contiguous_dim() {
        reduces_like_the_cpu(launch_pinned, sum(&[2, 3, 5000], 2));
    }

    #[test]
    fn sums_rows_along_the_contiguous_dim_unbound() {
        reduces_unbound_like_the_cpu(launch_zeroed_unbound, sum(&[2, 3, 5000], 2));
    }

    #[test]
    fn sums_rows_across_the_contiguous_dim() {
        reduces_like_the_cpu(launch_pinned, sum(&[3, 2100, 4], 1));
    }

    #[test]
    fn sums_rows_across_the_contiguous_dim_unbound() {
        reduces_unbound_like_the_cpu(launch_zeroed_unbound, sum(&[3, 2100, 4], 1));
    }

    /// Four rows, whose output's lie side by side in memory.
    #[test]
    fn sums_the_rows_of_a_matrix() {
        reduces_like_the_cpu(launch_pinned, sum(&[4, 9000], 1));
    }

    #[test]
    fn sums_the_rows_of_a_matrix_unbound() {
        reduces_unbound_like_the_cpu(launch_zeroed_unbound, sum(&[4, 9000], 1));
    }
}
