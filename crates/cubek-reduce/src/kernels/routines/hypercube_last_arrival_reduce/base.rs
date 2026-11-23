use super::kernel::{
    ReduceSplitExtents, ReduceSplitExtentsLaunch, hypercube_last_arrival_reduce_indexed_kernel,
    hypercube_last_arrival_reduce_kernel, hypercube_last_arrival_reduce_with_indices_kernel,
};
use crate::kernels::{
    PARTIALS, REDUCED, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem,
    ReduceWorkspaceBytes, WORKSPACE_TOO_SMALL, bound_at, each_microkernel_with, estimate_reduce,
    microkernel_widths, split_launch,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::LaunchArg;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Decline, TileOperand, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernelSettings;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row; the last of a row's cubes to finish merges the parts: few rows, too long
/// for one cube each to keep the device busy, as a sampler's argmax over a vocabulary.
///
/// Its cubes run along [`REDUCED`](crate::kernels::REDUCED), then [`BATCH`](crate::kernels::BATCH).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HypercubeLastArrivalReduceRoutine;

/// What a measurement settles for the last-arrival hypercube: each cube's microkernel, and the
/// lines a unit reads of a part, which set how many partials the last cube merges.
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
/// carrier of the operation's cells a part, each a line `width` wide.
fn partial_extent(problem: &ReduceRoutineProblem, part: usize, width: usize) -> usize {
    let cells = problem.reduced_extent().div_ceil(part) * problem.operation.written_per_row();
    cells * problem.row_values_a_line_at_width(width)
}

impl HypercubeLastArrivalReduceRoutine {
    /// The most scratch any of its settings asks of `problem`'s caller, at any width the device
    /// reads `problem`'s input at ([`ReduceWorkspace::bytes_for`](crate::kernels::ReduceWorkspace::bytes_for)).
    pub(crate) fn workspace_bound(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
    ) -> ReduceWorkspaceBytes {
        let widths = ladder_to(hw.vector_ceiling(problem.elements.input));
        Self::settings(hw)
            .iter()
            .filter_map(|settings| {
                HypercubeLastArrivalReduceMicrokernel::best(
                    hw,
                    &problem.work_item(),
                    &settings.microkernel,
                )
                .ok()
                .map(|microkernel| (microkernel.part().units(), settings.unit_lines))
            })
            .flat_map(|(units, unit_lines)| {
                widths.iter().map(move |&width| {
                    let part = problem.split_part(units, unit_lines, width);
                    workspace_at(problem, part, width)
                })
            })
            .fold(ReduceWorkspaceBytes::default(), ReduceWorkspaceBytes::max)
    }
}

/// The values of a row each of `blueprint`'s cubes reduces.
fn part_of(blueprint: &HypercubeLastArrivalReduceBlueprint) -> usize {
    blueprint
        .partitioning
        .box_of(ComputeScope::Cube)
        .extent(REDUCED)
}

/// The scratch `blueprint`'s launch of `problem` takes from its caller's workspace: the partials,
/// their indices and the counters of the rows in its cubes' parts.
fn workspace_of(
    problem: &ReduceRoutineProblem,
    blueprint: &HypercubeLastArrivalReduceBlueprint,
) -> ReduceWorkspaceBytes {
    workspace_at(problem, part_of(blueprint), blueprint.width)
}

/// Every power of two from one to `widest`.
fn ladder_to(widest: usize) -> Vec<usize> {
    core::iter::successors(Some(1), |width| Some(width * 2))
        .take_while(|width| *width <= widest)
        .collect()
}

/// The scratch a launch of `problem` takes in parts of `part` values, its lines `width` wide: the
/// partials of every box of rows in the accumulator's element, their indices in `u32` where its
/// carrier keeps any, and a `u32` counter a box of rows.
fn workspace_at(problem: &ReduceRoutineProblem, part: usize, width: usize) -> ReduceWorkspaceBytes {
    let cells: usize = problem
        .shape_with_reduced_extent(partial_extent(problem, part, width))
        .iter()
        .product();
    let index_bytes = size_of::<u32>() as u64;
    let indexed = problem.operation.carrier().carries_indices();
    ReduceWorkspaceBytes {
        partials: cells as u64 * problem.elements.accumulator.size() as u64,
        partial_indices: if indexed {
            cells as u64 * index_bytes
        } else {
            0
        },
        arrivals: problem.row_boxes_at_width(width) as u64 * index_bytes,
    }
}

impl Routine for HypercubeLastArrivalReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = HypercubeLastArrivalReduceMicrokernel;
    type Settings = HypercubeLastArrivalReduceSettings;
    type Blueprint = HypercubeLastArrivalReduceBlueprint;
    type Partitioning = Partitioning;

    /// Each hypercube microkernel setting crossed with
    /// [`UNIT_LINES`](HypercubeLastArrivalReduceSettings::UNIT_LINES).
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
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line())?;
        let (partitioning, grid) = split_launch(
            hw,
            problem,
            |part| {
                problem.split_space_with_partial_extent(partial_extent(problem, part, widths[0]))
            },
            microkernel.part().units(),
            settings.unit_lines,
            widths[0],
            microkernel.shared_memory_bytes_at_width(widths[0]),
        )?;
        let blueprint = ReduceBlueprint::new(microkernel, partitioning, widths);
        if !problem.workspace.holds(&workspace_of(problem, &blueprint)) {
            return Err(WORKSPACE_TOO_SMALL);
        }
        Ok((blueprint, grid))
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
        // unset; the counters, one a box of rows, hold zero, as the caller gave them and as every
        // launch leaves them. All of it is the caller's workspace, which prepare checked.
        let workspace = binding
            .workspace
            .clone()
            .expect("prepare declines a launch whose caller gave no workspace");
        let partial_extent = blueprint.partitioning.space().extent(PARTIALS);
        let partial_shape = problem.shape_with_reduced_extent(partial_extent);
        let counter = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::new_contiguous(
            partial_shape.clone(),
            workspace.partials,
            elements.accumulator,
        );
        let row_boxes = problem.row_boxes_at_width(blueprint.width);
        let arrivals = TensorHandle::new_contiguous([row_boxes], workspace.arrivals, counter);
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
        let partial_indices = problem.operation.carrier().carries_indices().then(|| {
            let handle =
                TensorHandle::new_contiguous(partial_shape, workspace.partial_indices, counter);
            bind_partials(handle, counter)
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
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// `problem` with as much scratch as it asks for.
    fn given_scratch(problem: ReduceRoutineProblem) -> ReduceRoutineProblem {
        ReduceRoutineProblem {
            workspace: ReduceWorkspaceBytes::UNBOUNDED,
            ..problem
        }
    }

    /// Four row axes, more than a launch grid has: every row axis is one [`BATCH`](crate::kernels::BATCH),
    /// so it prepares.
    #[test]
    fn prepares_more_row_axes_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = given_scratch(ReduceRoutineProblem::dense(
            &[2, 2, 2, 2, 64],
            4,
            ReduceOperation::ArgMax,
        ));
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

    /// Its caller gave no workspace: it declines, since cubek allocates none.
    #[test]
    fn declines_without_a_workspace() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let declined = HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings)
            .expect_err("no workspace was given");
        assert_eq!(declined.to_string(), WORKSPACE_TOO_SMALL.to_string());
    }

    /// The workspace `bytes_for` sizes holds what a launch at its last setting, its shortest
    /// parts and so its most partials, takes.
    #[test]
    fn the_bound_holds_the_shortest_parts_workspace() {
        let hw = devices::m2_pro();
        let problem = given_scratch(ReduceRoutineProblem::dense(
            &[3, 70001],
            1,
            ReduceOperation::ArgTopK { k: 3 },
        ));
        let settings = *HypercubeLastArrivalReduceRoutine::settings(&hw)
            .last()
            .expect("it lists settings");
        let (blueprint, _) = HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings)
            .expect("an M2 Pro serves it");
        let needs = workspace_of(&problem, &blueprint);
        let bound = HypercubeLastArrivalReduceRoutine::workspace_bound(&hw, &problem);
        assert!(bound.holds(&needs), "{bound:?} against {needs:?}");
    }
}
