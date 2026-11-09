use super::kernel::{
    ReduceSplit, ReduceSplitExtents, ReduceSplitExtentsLaunch, hypercube_reduce_kernel,
    hypercube_reduce_with_indices_kernel,
};
use crate::kernels::routines::cube::slot_bytes;
use crate::kernels::{
    ReduceBinding, ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors,
    reduce_widths,
};
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::reduce::hypercube::{
    HypercubeReduceMicrokernel, HypercubeReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrier, ReduceLayout, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row; the last of a row's cubes to finish merges the parts: few rows, too long
/// for one cube each to keep the device busy, as a sampler's argmax over a vocabulary.
///
/// Its cubes run along the reduced dim and at most two row dims; it declines more row dims.
#[derive(Debug, Clone, Copy, Default)]
pub struct HypercubeReduceRoutine;

/// The cube microkernel's planes, and how many lines each unit reads of a part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeReduceRoutineSettings {
    pub microkernel: HypercubeReduceMicrokernelSettings,
    pub unit_lines: usize,
}

impl HypercubeReduceRoutineSettings {
    /// Lines each unit reads of a part, the longest parts first: the fewest partials that still
    /// spread a row over the device.
    pub const UNIT_LINES: [usize; 3] = [32, 8, 2];
}

impl core::fmt::Display for HypercubeReduceRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} unit_lines={}", self.microkernel, self.unit_lines)
    }
}

pub type HypercubeReduceRoutineBlueprint = ReduceRoutineBlueprint<HypercubeReduceMicrokernel>;

impl ReduceSplit {
    /// Where the cubes of `problem`'s hypercube sit.
    pub fn new(problem: &ReduceRoutineProblem) -> Self {
        Self {
            reduced: problem.reduced(),
            rows: problem.rows(),
        }
    }
}

impl ReduceSplitExtents {
    /// The extents of `problem` split by `partitioning`, its owners `width` rows across the
    /// reduced dim.
    pub fn new(problem: &ReduceRoutineProblem, partitioning: &Partitioning, width: usize) -> Self {
        let part = partitioning
            .box_of(ComputeScope::Cube)
            .extent(problem.reduced());
        let inner_boxes = problem
            .row_boxes(width)
            .last()
            .map_or(1, |&(_, boxes)| boxes);
        Self {
            parts: to_u32(problem.reduced_extent().div_ceil(part)),
            values: to_u32(problem.reduced_extent()),
            inner_boxes: to_u32(inner_boxes),
        }
    }

    /// Boxes of rows of `problem`, its owners `width` rows across the reduced dim: each its cubes'
    /// counter.
    pub fn row_boxes(problem: &ReduceRoutineProblem, width: usize) -> usize {
        problem
            .row_boxes(width)
            .iter()
            .map(|&(_, boxes)| boxes)
            .product()
    }
}

fn to_u32(extent: usize) -> u32 {
    u32::try_from(extent).expect("a reduced extent fits in 32 bits")
}

/// Values of a box of rows' partials along the partials axis, for parts of `part` values: a
/// carrier of the operation's cells a part, each line `width` values of one row along the reduced
/// dim, one value of `width` rows across it.
fn partial_extent(problem: &ReduceRoutineProblem, part: usize, width: usize) -> usize {
    let cells = problem.reduced_extent().div_ceil(part) * problem.operation.written_per_row();
    match problem.layout() {
        ReduceLayout::Along => cells * width,
        ReduceLayout::Across => cells,
    }
}

/// Most row dims a hypercube's cubes run along, beside the reduced dim: a launch grid has three.
const MAX_ROW_DIMS: usize = 2;

impl Routine for HypercubeReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = HypercubeReduceMicrokernel;
    type Settings = HypercubeReduceRoutineSettings;
    type Blueprint = HypercubeReduceRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes, each with the longest parts first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        HypercubeReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                HypercubeReduceRoutineSettings::UNIT_LINES.map(|unit_lines| {
                    HypercubeReduceRoutineSettings {
                        microkernel,
                        unit_lines,
                    }
                })
            })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &HypercubeReduceMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ReduceOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if problem.rows().len() > MAX_ROW_DIMS {
            return Err(Refusal::Declined {
                why: "the hypercube reduce runs its cubes along at most two row dims",
            });
        }
        // The last cube merges a row's top-k lists one after another, `k` inserts a part, so the
        // shorter parts cost more in the merge than their extra cubes save.
        if matches!(
            problem.operation.carrier(),
            ReduceCarrier::TopK { .. } | ReduceCarrier::TopValues { .. }
        ) && settings.unit_lines < HypercubeReduceRoutineSettings::UNIT_LINES[0]
        {
            return Err(Refusal::Declined {
                why: "the hypercube splits a top-k's rows into its longest parts only",
            });
        }
        let microkernel =
            HypercubeReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widths = reduce_widths(hw, problem, vectorization);
        let units = microkernel.lanes * microkernel.planes;
        // Along, a line holds `width` values of the row; across, one.
        let part = units
            * settings.unit_lines
            * match problem.layout() {
                ReduceLayout::Along => widths[0],
                ReduceLayout::Across => 1,
            };
        let reduced = problem.reduced();
        let owner = [problem.owner_box(widths[0]).as_slice(), &[(reduced, part)]].concat();
        // The parts ride the grid's `X`, the boxes of rows its `Y` and `Z`.
        let cube_axes = [&[reduced], problem.rows().as_slice()].concat();
        let partitioning = Partitioning::new(
            problem.split_space(partial_extent(problem, part, widths[0])),
            Levels::leaf(&owner).cubes(&cube_axes).build(),
        );
        let cube = CubeResources {
            plane_width: units as u32,
            shared_memory: microkernel.planes * slot_bytes(problem, widths[0]),
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceRoutineBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
        &self,
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }
}

impl ReduceRoutine<HypercubeReduceMicrokernel> for HypercubeReduceRoutine {
    fn dispatch(
        &self,
        tensors: &ReduceTensors,
        binding: ReduceBinding,
        blueprint: HypercubeReduceRoutineBlueprint,
        grid: Grid,
    ) {
        let problem = tensors.problem();
        let elements = problem.elements;
        let client = tensors.client();
        let extents = ReduceSplitExtents::new(problem, &blueprint.partitioning, blueprint.width);
        // A box of rows' partials lie where its output does, the reduced dim's cells holding
        // every part's carrier; the counters, one a box of rows, hold zero.
        let partial_extent = blueprint.partitioning.space().extent(problem.partials());
        let partial_shape: Vec<usize> = (0..problem.shape.len())
            .map(|d| match d == problem.axis {
                true => partial_extent,
                false => problem.shape[d],
            })
            .collect();
        let counter = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::zeros(client, partial_shape.clone(), elements.accumulator);
        let partial_indices = TensorHandle::zeros(client, partial_shape, counter);
        let row_boxes = ReduceSplitExtents::row_boxes(problem, blueprint.width);
        let arrivals = TensorHandle::zeros(client, [row_boxes], counter);
        let bind_partials = |handle: TensorHandle, element| {
            binding
                .launcher
                .arg(handle.binding(), element)
                .axes(&problem.partial_axes())
                .vectorize(blueprint.width)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
        };
        let partials = bind_partials(partials, elements.accumulator);
        let partial_indices = bind_partials(partial_indices, counter);
        let arrivals = arrivals.binding().into_tensor_arg().into_buffer_arg();
        // SAFETY: every operand but the counters is bound through the launcher, which masks the
        // reads and writes of a tile the problem's edge cuts short, and the kernel indexes no
        // buffer but through its tiles and the counters, one a box of rows, which the cubes count
        // themselves in at by their box's coordinates. A cube writes no partial but its own part's,
        // and the last cube of a box reads them after every cube of the box stored its own. Its
        // shared memory is the slots its microkernel declares, which the grid counted.
        match (binding.indices, problem.indices) {
            (Some(indices), Some(index)) => unsafe {
                hypercube_reduce_with_indices_kernel::launch_unchecked(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input.arg(),
                    binding.out.arg(),
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
                    ReduceSplit::new(problem),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out, index],
                )
            },
            _ => unsafe {
                hypercube_reduce_kernel::launch_unchecked(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input.arg(),
                    binding.out.arg(),
                    partials.arg(),
                    partial_indices.arg(),
                    arrivals,
                    binding.launcher.partitioning_arg(),
                    ReduceSplitExtentsLaunch::new(
                        extents.parts,
                        extents.values,
                        extents.inner_boxes,
                    ),
                    ReduceSplit::new(problem),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out],
                )
            },
        }
    }
}
