use super::kernel::hypercube_atomic_reduce_kernel;
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem, each_microkernel_with,
    estimated_work, split_launch,
};
use cubecl::client::Client;
use cubecl::ir::Type;
use cubecl::ir::features::AtomicUsage;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{AccumulateArgLaunch, Bound, Decline, TileOperand, Vectorization};
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernelSettings;
use cubek_microkernel::reduce::hypercube::HypercubeAtomicReduceMicrokernel;
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
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

pub type HypercubeAtomicReduceBlueprint = ReduceBlueprint<HypercubeAtomicReduceMicrokernel>;

impl Routine for HypercubeAtomicReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = HypercubeAtomicReduceMicrokernel;
    type Settings = HypercubeAtomicReduceSettings;
    type Blueprint = HypercubeAtomicReduceBlueprint;
    type Partitioning = ();

    /// Each cube microkernel setting crossed with
    /// [`UNIT_LINE_COUNTS`](HypercubeAtomicReduceSettings::UNIT_LINE_COUNTS).
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        each_microkernel_with(
            HypercubeAtomicReduceMicrokernel::settings(hw),
            HypercubeAtomicReduceSettings::UNIT_LINE_COUNTS,
            |microkernel, unit_line_count| HypercubeAtomicReduceSettings {
                microkernel,
                unit_line_count,
            },
        )
    }

    fn vectorize(
        microkernel: &HypercubeAtomicReduceMicrokernel,
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
        if !hw
            .properties
            .atomic_type_usage(Type::atomic(problem.elements.out))
            .contains(AtomicUsage::Add)
        {
            return Err(Decline::Unserved {
                why: "the device has no atomic add for the output's element",
            });
        }
        let microkernel = HypercubeAtomicReduceMicrokernel::best(
            hw,
            &problem.work_item(),
            &settings.microkernel,
        )?;
        let part = microkernel.part();
        let lane_count = Self::lane_count(hw, problem, &microkernel, part.widest_line())?;
        let cube = CubeResources {
            plane_width: part.units() as u32,
            shared_memory: part.shared_memory_bytes_at_lane_count(lane_count),
        };
        let (partitioning, grid) = split_launch(
            hw,
            problem,
            |_| problem.space(),
            cube,
            settings.unit_line_count,
            lane_count,
        )?;
        Ok((
            ReduceBlueprint {
                microkernel,
                partitioning,
                lane_count,
            },
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

impl ReduceRoutine<HypercubeAtomicReduceMicrokernel> for HypercubeAtomicReduceRoutine {
    const NAME: &'static str = "hypercube-atomic-reduce";

    /// The output as the launch binds it, which its kernel takes as an atomic argument.
    type Out = Bound;

    fn out_arg(out: Bound) -> Bound {
        out
    }

    /// A sum accumulates into values alone, whichever carrier `F` serves.
    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, Bound>,
        blueprint: HypercubeAtomicReduceBlueprint,
        grid: Grid,
    ) -> Result<(), Decline> {
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::{
        BufferScopeChecker, Reduction, UnboundScopeChecker, bound_back, unbound_tensors,
    };
    use crate::kernels::{
        ReduceKernel, ReduceTensors, launch_kernel, launch_zeroed_atomic_unbound,
    };
    use cubecl::prelude::TensorBinding;
    use cubecl::tile::launch::StoredTileOperand;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernel;
    use cubek_std::devices;
    use cubek_test_utils::ScopeChecker;

    declines_lacking_every_feature!(HypercubeAtomicReduceRoutine);
    names_its_setting!(
        HypercubeAtomicReduceRoutine,
        "planes=4 lane_slots=32 unit_lines=32"
    );
    estimates_its_microkernel!(
        HypercubeAtomicReduceRoutine,
        ReduceRoutineProblem::work_item
    );

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
        BufferScopeChecker::new(launch_pinned).check([sum(&[1, 1, 9000], 2)]);
    }

    #[test]
    fn sums_one_long_row_unbound() {
        UnboundScopeChecker::new(launch_zeroed_unbound).check([sum(&[1, 1, 9000], 2)]);
    }

    #[test]
    fn sums_rows_along_the_contiguous_dim() {
        BufferScopeChecker::new(launch_pinned).check([sum(&[2, 3, 5000], 2)]);
    }

    #[test]
    fn sums_rows_along_the_contiguous_dim_unbound() {
        UnboundScopeChecker::new(launch_zeroed_unbound).check([sum(&[2, 3, 5000], 2)]);
    }

    #[test]
    fn sums_rows_across_the_contiguous_dim() {
        BufferScopeChecker::new(launch_pinned).check([sum(&[3, 2100, 4], 1)]);
    }

    #[test]
    fn sums_rows_across_the_contiguous_dim_unbound() {
        UnboundScopeChecker::new(launch_zeroed_unbound).check([sum(&[3, 2100, 4], 1)]);
    }

    /// Four rows, whose output's lie side by side in memory.
    #[test]
    fn sums_the_rows_of_a_matrix() {
        BufferScopeChecker::new(launch_pinned).check([sum(&[4, 9000], 1)]);
    }

    #[test]
    fn sums_the_rows_of_a_matrix_unbound() {
        UnboundScopeChecker::new(launch_zeroed_unbound).check([sum(&[4, 9000], 1)]);
    }
}
