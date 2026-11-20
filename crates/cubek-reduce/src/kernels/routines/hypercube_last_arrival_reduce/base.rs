use super::kernel::{
    ReduceSplitExtents, ReduceSplitExtentsLaunch, hypercube_last_arrival_reduce_indexed_kernel,
    hypercube_last_arrival_reduce_kernel, hypercube_last_arrival_reduce_with_indices_kernel,
};
use crate::kernels::{
    BATCH, PARTIALS, REDUCED, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem,
    bound_at, each_microkernel_with, estimate_reduce, microkernel_widths,
};
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::LaunchArg;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Decline, TileOperand, Vectorization};
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernelSettings;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row; the last of a row's cubes to finish merges the parts: few rows, too long
/// for one cube each to keep the device busy, as a sampler's argmax over a vocabulary.
///
/// Its cubes run along [`REDUCED`], then [`BATCH`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HypercubeLastArrivalReduceRoutine;

/// The cube microkernel's planes, and how many lines each unit reads of a part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalReduceSettings {
    pub microkernel: CubeStagedReduceMicrokernelSettings,
    pub unit_lines: usize,
}

impl HypercubeLastArrivalReduceSettings {
    /// Lines each unit reads of a part, the longest parts first: the fewest partials that still
    /// spread a row over the device.
    pub const UNIT_LINES: [usize; 3] = [32, 8, 2];
}

impl core::fmt::Display for HypercubeLastArrivalReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} unit_lines={}", self.microkernel, self.unit_lines)
    }
}

pub type HypercubeLastArrivalReduceBlueprint =
    ReduceBlueprint<HypercubeLastArrivalReduceMicrokernel>;

/// Values of a box of rows' partials along the partials dimension, for parts of `part` values: a
/// carrier of the operation's cells a part, each line `width` values of one row along the reduced
/// dim, one value of `width` rows across it.
fn partial_extent(problem: &ReduceRoutineProblem, part: usize, width: usize) -> usize {
    let cells = problem.reduced_extent().div_ceil(part) * problem.operation.written_per_row();
    cells * problem.input_load_at_width(width).extent_along(REDUCED)
}

impl Routine for HypercubeLastArrivalReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = HypercubeLastArrivalReduceMicrokernel;
    type Settings = HypercubeLastArrivalReduceSettings;
    type Blueprint = HypercubeLastArrivalReduceBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes, each with the longest parts first.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        each_microkernel_with(
            HypercubeLastArrivalReduceMicrokernel::settings(hw),
            HypercubeLastArrivalReduceSettings::UNIT_LINES,
            |microkernel, unit_lines| HypercubeLastArrivalReduceSettings {
                microkernel,
                unit_lines,
            },
        )
    }

    fn vectorize(
        microkernel: &HypercubeLastArrivalReduceMicrokernel,
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
        let microkernel = HypercubeLastArrivalReduceMicrokernel::best(
            hw,
            &problem.work_item(),
            &settings.microkernel,
        )?;
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line());
        let units = microkernel.part().units();
        let part = problem.split_part(units, settings.unit_lines, widths[0]);
        let owner = problem.split_owner_box(widths[0], part);
        // The parts ride the grid's `X`, the boxes of rows the rest.
        let partitioning = Partitioning::new(
            problem.split_space_with_partial_extent(partial_extent(problem, part, widths[0])),
            Levels::leaf(&owner).cubes(&[REDUCED, BATCH]).build(),
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

impl ReduceLauncher<HypercubeLastArrivalReduceMicrokernel> for HypercubeLastArrivalReduceRoutine {
    const NAME: &'static str = "hypercube-last-arrival-reduce";

    fn dispatch<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: HypercubeLastArrivalReduceBlueprint,
        grid: Grid,
    ) {
        let elements = problem.elements;
        let extents = ReduceSplitExtents::new(problem, &blueprint.partitioning);
        // A box of rows' partials lie where its output does, the reduced dim's cells holding
        // every part's carrier, each stored by its part's cube before any is read, so they start
        // unset; the counters, one a box of rows, start at zero, copied from the host rather than
        // set by a launch of their own.
        let partial_extent = blueprint.partitioning.space().extent(PARTIALS);
        let partial_shape = problem.shape_with_reduced_extent(partial_extent);
        let counter = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::empty(client, partial_shape.clone(), elements.accumulator);
        let row_boxes = problem.row_boxes_at_width(blueprint.width);
        let arrivals = TensorHandle::new_contiguous(
            [row_boxes],
            client.create(Bytes::from_elems(vec![0u32; row_boxes])),
            counter,
        );
        let bind_partials = |handle: TensorHandle, element| {
            bound_at(
                &binding.launcher,
                problem,
                handle.binding(),
                PARTIALS,
                element,
                blueprint.width,
            )
        };
        let partials = bind_partials(partials, elements.accumulator);
        // The partials' indices, only where the carrier keeps any.
        let partial_indices =
            problem.operation.carrier().carries_indices().then(|| {
                bind_partials(TensorHandle::empty(client, partial_shape, counter), counter)
            });
        let arrivals = arrivals.binding().into_tensor_arg().into_buffer_arg();
        // SAFETY: every operand but the counters is bound through the launcher, which masks the
        // reads and writes of a tile the problem's edge cuts short, and the kernel indexes no
        // buffer but through its tiles and the counters, one a box of rows, which the cubes count
        // themselves in at by their box's coordinates. A cube writes no partial but its own part's,
        // and the last cube of a box reads them after every cube of the box stored its own. Its
        // shared memory is the slots its microkernel declares, which the grid counted.
        let written = partial_indices
            .map(|partial_indices| (partial_indices, binding.indices.zip(problem.indices)));
        unsafe {
            match written {
                Some((partial_indices, Some((indices, index)))) => {
                    hypercube_last_arrival_reduce_with_indices_kernel::launch_unchecked::<I>(
                        client,
                        grid.launch.cube_count,
                        grid.launch.cube_dim,
                        blueprint.width,
                        binding.input,
                        binding.out,
                        indices.arg(),
                        partials.arg(),
                        partial_indices.arg(),
                        arrivals,
                        binding.launcher.partitioning_arg(),
                        ReduceSplitExtentsLaunch::new(extents.parts, extents.values),
                        blueprint.microkernel,
                        [
                            elements.input,
                            elements.accumulator,
                            elements.out,
                            index.element,
                        ],
                    )
                }
                Some((partial_indices, None)) => {
                    hypercube_last_arrival_reduce_indexed_kernel::launch_unchecked::<I>(
                        client,
                        grid.launch.cube_count,
                        grid.launch.cube_dim,
                        blueprint.width,
                        binding.input,
                        binding.out,
                        partials.arg(),
                        partial_indices.arg(),
                        arrivals,
                        binding.launcher.partitioning_arg(),
                        ReduceSplitExtentsLaunch::new(extents.parts, extents.values),
                        blueprint.microkernel,
                        [elements.input, elements.accumulator, elements.out],
                    )
                }
                None => hypercube_last_arrival_reduce_kernel::launch_unchecked::<I>(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input,
                    binding.out,
                    partials.arg(),
                    arrivals,
                    binding.launcher.partitioning_arg(),
                    ReduceSplitExtentsLaunch::new(extents.parts, extents.values),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out],
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::routines::routine_check;
    use cubecl::tile::launch::Decline;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Four row axes, more than a launch grid has: every row axis is one [`BATCH`](crate::kernels::BATCH),
    /// so it prepares.
    #[test]
    fn prepares_more_row_axes_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, ReduceOperation::ArgMax);
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let prepared =
            HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }

    /// tracel:kernel::routine-tested-without-features: a Decline.
    #[test]
    fn declines_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let prepared =
            HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
    }

    /// Its cubes meet through their stores, so without device memory scope alone it declines.
    #[test]
    fn declines_without_device_memory_scope() {
        let hw = {
            let mut hw = devices::m2_pro();
            devices::features(&mut hw).device_memory_scope = false;
            hw
        };
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::ArgMax);
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let prepared =
            HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
    }

    /// The routine at the last setting it serves, its shortest parts, so that rows too short for
    /// its first setting's parts still split, launched from `input` into `out`.
    fn launch_at_shortest_parts(
        client: &cubecl::client::Client,
        reduction: &routine_check::Reduction,
        input: cubecl::prelude::TensorBinding,
        out: cubecl::prelude::TensorBinding,
    ) -> Result<(), Decline> {
        let hw = cubek_std::Hardware::from_client(client);
        let tensors = reduction.tensors(client, input, out);
        let (blueprint, grid) = HypercubeLastArrivalReduceRoutine::settings(&hw)
            .iter()
            .rev()
            .map(|settings| {
                HypercubeLastArrivalReduceRoutine::prepare(&hw, tensors.problem(), settings)
            })
            .reduce(|last, earlier| last.or(earlier))
            .unwrap_or(Err(cubek_std::routine::ROUTINE_LISTS_NO_SETTING))?;
        HypercubeLastArrivalReduceRoutine::launch(&tensors, blueprint, grid);
        Ok(())
    }

    /// A test for each operation whose rows hold what it writes, each one launch at the shortest
    /// parts of `shape` reduced along `axis`.
    macro_rules! every_operation {
        ($shape:expr, $axis:expr) => {
            routine_check::reduction_tests!(super::launch_at_shortest_parts, $shape, $axis, [
                sum: ReduceOperation::Sum,
                prod: ReduceOperation::Prod,
                mean: ReduceOperation::Mean,
                max: ReduceOperation::Max,
                min: ReduceOperation::Min,
                max_abs: ReduceOperation::MaxAbs,
                any: ReduceOperation::Any,
                all: ReduceOperation::All,
                arg_max: ReduceOperation::ArgMax,
                arg_min: ReduceOperation::ArgMin,
                top_1: ReduceOperation::TopK { k: 1 },
                arg_top_1: ReduceOperation::ArgTopK { k: 1 },
                top_3: ReduceOperation::TopK { k: 3 },
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        };
    }

    /// A `[2, 3, 5000]` tensor along its contiguous dim, in three parts.
    mod along_the_contiguous_dim {
        use super::*;
        every_operation!([2, 3, 5000], 2);
    }

    /// A `[3, 2100, 4]` tensor along its middle dim, across the contiguous one, in five parts.
    mod across_the_contiguous_dim {
        use super::*;
        every_operation!([3, 2100, 4], 1);
    }

    /// Rows three long, of one part.
    mod rows_of_one_part {
        use super::*;
        every_operation!([2, 3, 5000], 1);
    }

    /// More row axes than a grid has, the rows on both sides of the reduced axis.
    mod more_row_axes_around_the_reduced_one {
        use super::*;
        routine_check::reduction_tests!(super::launch_at_shortest_parts, [2, 3, 2, 2, 64], 2, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            arg_max: ReduceOperation::ArgMax,
        ]);
    }

    /// More row axes than a grid has, all before the reduced axis.
    mod more_row_axes_before_the_reduced_one {
        use super::*;
        routine_check::reduction_tests!(super::launch_at_shortest_parts, [2, 3, 2, 2, 64], 4, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            arg_max: ReduceOperation::ArgMax,
        ]);
    }
}
