use super::kernel::{
    MAX_SPLIT_ROW_DIMS, ReduceSplitExtents, ReduceSplitExtentsLaunch,
    hypercube_last_arrival_reduce_indexed_kernel, hypercube_last_arrival_reduce_kernel,
    hypercube_last_arrival_reduce_with_indices_kernel,
};
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, estimate_reduce,
    microkernel_widths,
};
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::LaunchArg;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Decline, TileOperand, Vectorization};
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernelSettings;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row; the last of a row's cubes to finish merges the parts: few rows, too long
/// for one cube each to keep the device busy, as a sampler's argmax over a vocabulary.
///
/// Its cubes run along the reduced dim and at most two row dims; it declines more row dims.
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
    match problem.layout() {
        ReduceLayout::Along => cells * width,
        ReduceLayout::Across => cells,
    }
}

impl Routine for HypercubeLastArrivalReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = HypercubeLastArrivalReduceMicrokernel;
    type Settings = HypercubeLastArrivalReduceSettings;
    type Blueprint = HypercubeLastArrivalReduceBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes, each with the longest parts first.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        HypercubeLastArrivalReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                HypercubeLastArrivalReduceSettings::UNIT_LINES.map(|unit_lines| {
                    HypercubeLastArrivalReduceSettings {
                        microkernel,
                        unit_lines,
                    }
                })
            })
            .collect()
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
        if problem.rows().len() > MAX_SPLIT_ROW_DIMS {
            return Err(Decline::Unserved {
                why: "the hypercube reduce runs its cubes along at most two row dims",
            });
        }
        let microkernel = HypercubeLastArrivalReduceMicrokernel::best(
            hw,
            &problem.work_item(),
            &settings.microkernel,
        )?;
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line());
        let units = microkernel.part().units();
        let part = problem.split_part(units, settings.unit_lines, widths[0]);
        let owner = problem.split_owner_box(widths[0], part);
        let cube_dimensions = problem.split_cube_dimensions();
        let partitioning = Partitioning::new(
            problem.split_space_with_partial_extent(partial_extent(problem, part, widths[0])),
            Levels::leaf(&owner).cubes(&cube_dimensions).build(),
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
        let extents = ReduceSplitExtents::new(problem, &blueprint.partitioning, blueprint.width);
        // A box of rows' partials lie where its output does, the reduced dim's cells holding
        // every part's carrier, each stored by its part's cube before any is read, so they start
        // unset; the counters, one a box of rows, start at zero, copied from the host rather than
        // set by a launch of their own.
        let partial_extent = blueprint.partitioning.space().extent(problem.partials());
        let partial_shape: Vec<usize> = (0..problem.shape.len())
            .map(|d| match d == problem.axis {
                true => partial_extent,
                false => problem.shape[d],
            })
            .collect();
        let counter = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::empty(client, partial_shape.clone(), elements.accumulator);
        let row_boxes = ReduceSplitExtents::row_boxes(problem, blueprint.width);
        let arrivals = TensorHandle::new_contiguous(
            [row_boxes],
            client.create(Bytes::from_elems(vec![0u32; row_boxes])),
            counter,
        );
        let bind_partials = |handle: TensorHandle, element| {
            binding
                .launcher
                .arg(handle.binding(), element)
                .dimensions(&problem.partial_dimensions())
                .vectorize(blueprint.width)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
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
                        ReduceSplitExtentsLaunch::new(
                            extents.parts,
                            extents.values,
                            extents.inner_boxes,
                        ),
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
                        ReduceSplitExtentsLaunch::new(
                            extents.parts,
                            extents.values,
                            extents.inner_boxes,
                        ),
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
                    ReduceSplitExtentsLaunch::new(
                        extents.parts,
                        extents.values,
                        extents.inner_boxes,
                    ),
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
    use cubecl::tile::launch::Decline;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Four row dims, more than a launch grid has: a Decline, not a panic.
    #[test]
    fn declines_more_row_dims_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, ReduceOperation::ArgMax);
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let prepared =
            HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
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
}
