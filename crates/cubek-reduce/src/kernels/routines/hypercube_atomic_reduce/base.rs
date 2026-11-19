use super::kernel::hypercube_atomic_reduce_kernel;
use crate::kernels::{
    MAX_SPLIT_ROW_DIMS, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem,
    ReduceTensors, UNBOUND_LINES_NARROWER, estimate_reduce, microkernel_widths,
};
use core::convert::identity;
use cubecl::client::Client;
use cubecl::ir::Type;
use cubecl::ir::features::AtomicUsage;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{
    AccumulateArgLaunch, Bound, Decline, StoredTileOperand, TileOperand, Unbound, Vectorization,
};
use cubecl::tile::{Geometry, Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceOperand, ReduceOperation};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row and adds its total into the output: a sum of few rows, too long for one
/// cube each to keep the device busy, as a whole tensor summed into one value.
///
/// The output must hold zero before it runs ([`UnitDirectZeroReduceRoutine`](crate::kernels::UnitDirectZeroReduceRoutine)
/// zeroes it), since it only ever adds. Its cubes run along the reduced
/// dim and at most two row dims; it declines more.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HypercubeAtomicReduceRoutine;

/// The cube microkernel's planes, and how many lines each unit reads of a part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAtomicReduceSettings {
    pub microkernel: CubeStagedReduceMicrokernelSettings,
    pub unit_lines: usize,
}

impl HypercubeAtomicReduceSettings {
    /// Lines each unit reads of a part, the longest parts first: the fewest atomic adds that still
    /// spread a row over the device.
    pub const UNIT_LINES: [usize; 3] = [32, 8, 2];
}

impl core::fmt::Display for HypercubeAtomicReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} unit_lines={}", self.microkernel, self.unit_lines)
    }
}

pub type HypercubeAtomicReduceBlueprint = ReduceBlueprint<CubeStagedReduceMicrokernel>;

impl Routine for HypercubeAtomicReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = CubeStagedReduceMicrokernel;
    type Settings = HypercubeAtomicReduceSettings;
    type Blueprint = HypercubeAtomicReduceBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes, each with the longest parts first.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                HypercubeAtomicReduceSettings::UNIT_LINES.map(|unit_lines| {
                    HypercubeAtomicReduceSettings {
                        microkernel,
                        unit_lines,
                    }
                })
            })
            .collect()
    }

    fn vectorize(
        microkernel: &CubeStagedReduceMicrokernel,
        _partitioning: &Partitioning,
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
        if problem.rows().len() > MAX_SPLIT_ROW_DIMS {
            return Err(Decline::Unserved {
                why: "the atomic hypercube runs its cubes along at most two row dims",
            });
        }
        let microkernel =
            CubeStagedReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line());
        let units = microkernel.units();
        let part = problem.split_part(units, settings.unit_lines, widths[0]);
        let owner = problem.split_owner_box(widths[0], part);
        let cube_axes = problem.split_cube_dimensions();
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&owner).cubes(&cube_axes).build(),
        );
        let cube = CubeResources {
            plane_width: units as u32,
            shared_memory: microkernel.shared_memory_bytes_at_width(widths[0]),
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        estimate_reduce(problem, &blueprint.microkernel, grid, estimate);
    }
}

impl ReduceLauncher<CubeStagedReduceMicrokernel> for HypercubeAtomicReduceRoutine {
    const NAME: &'static str = "hypercube-atomic-reduce";

    /// Never reached: [`launch`](Self::launch) and [`launch_unbound`](Self::launch_unbound) bind
    /// the output as the atomic argument the kernel adds into, which a plain output tensor is not.
    fn dispatch<I: TileOperand>(
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
    ///
    /// # Errors
    ///
    /// [`UNBOUND_LINES_NARROWER`] where `blueprint` reads narrower lines than the caller's input is
    /// stated at.
    fn launch_unbound<I: TileOperand>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: HypercubeAtomicReduceBlueprint,
        grid: Grid,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let [input_lines, _] = tensors.problem().elem_alignments;
        if !input_lines.admits(blueprint.width) {
            return Err(UNBOUND_LINES_NARROWER);
        }
        let binding = tensors.bind(&blueprint, &grid).map(input, identity);
        launch_atomic::<I>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
        Ok(())
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
    let (spec, width) = (binding.out.spec.clone(), binding.out.vector_size);
    let out = AccumulateArgLaunch::new(binding.out.tensor(), width, spec);
    // SAFETY: both operands are bound through the launcher, which masks the reads and writes of a
    // tile the problem's edge cuts short, and the kernel indexes no buffer but through its tiles;
    // the output only receives atomic adds. Its shared memory is the slots its microkernel
    // declares, which the grid counted.
    unsafe {
        hypercube_atomic_reduce_kernel::launch_unchecked::<I>(
            client,
            grid.launch.cube_count,
            grid.launch.cube_dim,
            blueprint.width,
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
    use cubecl::tile::launch::Decline;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// tracel:kernel::routine-tested-without-features: no atomics, so a Decline.
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
}
