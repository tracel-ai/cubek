use super::kernel::{
    ReduceSplitExtents, ReduceSplitExtentsLaunch, hypercube_last_arrival_reduce_kernel,
};
use crate::kernels::{
    PARTIALS, REDUCED, ReduceBinding, ReduceBlueprint, ReducePartials, ReduceRoutine,
    ReduceRoutineProblem, bound_at, each_microkernel_with, estimated_work, split_launch,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::LaunchArg;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{Bound, Decline, TileOperand, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernelSettings;
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every row split into parts along the reduced dim, a cube a part, which reduces it as the cube
/// routine reduces a row; the last of a row's cubes to finish merges the parts: few rows, too long
/// for one cube each to keep the device busy, as a sampler's argmax over a vocabulary.
///
/// Its cubes run along [`REDUCED`], then [`BATCH`](crate::kernels::BATCH).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HypercubeLastArrivalReduceRoutine;

/// What a measurement settles for the last-arrival hypercube: each cube's microkernel, and the
/// lines a unit reads of a part, which set how many partials the last cube merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalReduceSettings {
    pub microkernel: CubeStagedReduceMicrokernelSettings,
    pub unit_line_count: usize,
}

impl HypercubeLastArrivalReduceSettings {
    /// Lines each unit reads of a part, the longest parts first: the fewest partials that still
    /// spread a row over the device.
    pub const UNIT_LINE_COUNTS: [usize; 3] = [32, 8, 2];
}

impl core::fmt::Display for HypercubeLastArrivalReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} unit_lines={}",
            self.microkernel, self.unit_line_count
        )
    }
}

pub type HypercubeLastArrivalReduceBlueprint =
    ReduceBlueprint<HypercubeLastArrivalReduceMicrokernel>;

/// Values of a box of rows' partials along the partials dimension, for parts of `part` values: a
/// carrier of the operation's cells a part, each a line `lane_count` wide.
fn partial_count(problem: &ReduceRoutineProblem, part: usize, lane_count: usize) -> usize {
    let cells = problem.reduced_extent().div_ceil(part) * problem.operation.written_per_row();
    cells * problem.row_value_count_at_lane_count(lane_count)
}

impl HypercubeLastArrivalReduceRoutine {
    /// The most partials any of its settings writes of `problem`, at any lanes the device reads
    /// `problem`'s input at ([`ReduceWorkspace::bytes_for`](crate::kernels::ReduceWorkspace::bytes_for)).
    pub(crate) fn partials_bound(hw: &Hardware, problem: &ReduceRoutineProblem) -> ReducePartials {
        let lane_counts = lane_counts_up_to_widest_count(hw.vector_ceiling(problem.elements.input));
        Self::settings(hw)
            .iter()
            .filter_map(|settings| {
                HypercubeLastArrivalReduceMicrokernel::best(
                    hw,
                    &problem.work_item(),
                    &settings.microkernel,
                )
                .ok()
                .map(|microkernel| (microkernel.part().units(), settings.unit_line_count))
            })
            .flat_map(|(units, unit_lines)| {
                lane_counts.iter().map(move |&lane_count| {
                    let part = problem.split_part(units, unit_lines, lane_count);
                    partials_at(problem, part, lane_count)
                })
            })
            .fold(ReducePartials::default(), ReducePartials::max)
    }
}

/// The partials `blueprint`'s launch of `problem` writes: those of the rows in its cubes' parts.
fn partials_of(
    problem: &ReduceRoutineProblem,
    blueprint: &HypercubeLastArrivalReduceBlueprint,
) -> ReducePartials {
    let part = blueprint
        .partitioning
        .box_of(ComputeScope::Cube)
        .extent(REDUCED);
    partials_at(problem, part, blueprint.lane_count)
}

/// Every power of two from one to `widest_count`.
fn lane_counts_up_to_widest_count(widest_count: usize) -> Vec<usize> {
    const NARROWEST_COUNT: usize = 1;
    core::iter::successors(Some(NARROWEST_COUNT), |lane_count| Some(lane_count * 2))
        .take_while(|lane_count| *lane_count <= widest_count)
        .collect()
}

/// The partials a launch of `problem` writes in parts of `part` values, its lines `lane_count` wide:
/// those of every box of rows, their indices where its carrier keeps any, and a count of arrived
/// parts a box of rows.
fn partials_at(problem: &ReduceRoutineProblem, part: usize, lane_count: usize) -> ReducePartials {
    let value_count: usize = problem
        .shape_with_reduced_count(partial_count(problem, part, lane_count))
        .iter()
        .product();
    ReducePartials {
        value_count,
        index_count: match problem.operation.carrier().carries_indices() {
            true => value_count,
            false => 0,
        },
        arrival_count: problem.row_box_count_at_lane_count(lane_count),
    }
}

/// Why a problem whose caller gave no buffers for its partials declines: the routine writes them.
const WRITES_NO_PARTIALS: Decline = Decline::Unserved {
    why: "the problem writes no partials, which the last-arrival hypercube's parts meet through",
};

/// Why a problem whose partials hold fewer cells than its rows' parts declines.
const PARTIALS_TOO_SHORT: Decline = Decline::Unserved {
    why: "the partials' buffers hold fewer cells than the parts the rows are split into",
};

/// Why a row of more values than a `u32` counts declines: the kernel counts them in one.
const ROW_PAST_32_BITS: Decline = Decline::Unserved {
    why: "a row split across cubes holds more values than 32 bits count",
};

impl Routine for HypercubeLastArrivalReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = HypercubeLastArrivalReduceMicrokernel;
    type Settings = HypercubeLastArrivalReduceSettings;
    type Blueprint = HypercubeLastArrivalReduceBlueprint;
    type Partitioning = ();

    /// Each hypercube microkernel setting crossed with
    /// [`UNIT_LINE_COUNTS`](HypercubeLastArrivalReduceSettings::UNIT_LINE_COUNTS).
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        each_microkernel_with(
            HypercubeLastArrivalReduceMicrokernel::settings(hw),
            HypercubeLastArrivalReduceSettings::UNIT_LINE_COUNTS,
            |microkernel, unit_line_count| HypercubeLastArrivalReduceSettings {
                microkernel,
                unit_line_count,
            },
        )
    }

    fn vectorize(
        microkernel: &HypercubeLastArrivalReduceMicrokernel,
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
        let (microkernel, lane_count) =
            best_microkernel!(HypercubeLastArrivalReduceMicrokernel, hw, problem, settings);
        let cube = CubeResources {
            plane_width: microkernel.part().units() as u32,
            shared_memory: microkernel.shared_memory_bytes_at_lane_count(lane_count),
        };
        let (partitioning, grid) = split_launch(
            hw,
            problem,
            |part| problem.split_space_with_partial_count(partial_count(problem, part, lane_count)),
            cube,
            settings.unit_line_count,
            lane_count,
        )?;
        let blueprint = ReduceBlueprint {
            microkernel,
            partitioning,
            lane_count,
        };
        if u32::try_from(problem.reduced_extent()).is_err() {
            return Err(ROW_PAST_32_BITS);
        }
        let partials = problem.partials.ok_or(WRITES_NO_PARTIALS)?;
        if !partials.holds(&partials_of(problem, &blueprint)) {
            return Err(PARTIALS_TOO_SHORT);
        }
        Ok((blueprint, grid))
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

impl ReduceRoutine<HypercubeLastArrivalReduceMicrokernel> for HypercubeLastArrivalReduceRoutine {
    const NAME: &'static str = "hypercube-last-arrival-reduce";

    type Out = TileTensorLaunch;

    fn out_arg(out: Bound) -> TileTensorLaunch {
        out.arg()
    }

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: HypercubeLastArrivalReduceBlueprint,
        grid: Grid,
    ) {
        let extents = ReduceSplitExtents::new(problem, &blueprint.partitioning);
        // A box of rows' partials lie where its output does, the reduced dim's cells holding
        // every part's carrier, each stored by its part's cube before any is read, so they start
        // unset; the counters, one a box of rows, hold zero, as the caller gave them and as every
        // launch leaves them. All of it is the caller's workspace, which prepare checked.
        let workspace = binding
            .workspace
            .clone()
            .expect("prepare declines a launch whose caller gave no workspace");
        let partial_count = blueprint.partitioning.space().extent(PARTIALS);
        let partial_shape = problem.shape_with_reduced_count(partial_count);
        let counter = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::new_contiguous(
            partial_shape.clone(),
            workspace.partials,
            problem.elements.accumulator,
        );
        let row_boxes = problem.row_box_count_at_lane_count(blueprint.lane_count);
        let arrivals = TensorHandle::new_contiguous([row_boxes], workspace.arrivals, counter);
        let bind_partials = |handle: TensorHandle, element| {
            bound_at(
                &binding.launcher,
                problem,
                handle.binding(),
                PARTIALS,
                element,
                blueprint.lane_count,
            )
        };
        let partials = bind_partials(partials, problem.elements.accumulator);
        // The partials' indices, only where the carrier keeps any.
        let partial_indices = problem.operation.carrier().carries_indices().then(|| {
            let handle =
                TensorHandle::new_contiguous(partial_shape, workspace.partial_indices, counter);
            bind_partials(handle, counter)
        });
        let arrivals = arrivals.binding().into_tensor_arg().into_buffer_arg();
        // Its counters are indexed past its tiles, one a box of rows, which the cubes count
        // themselves in at by their box's coordinates. A cube writes no partial but its own part's,
        // and the last cube of a box reads them after every cube of the box stored its own.
        launch_reduce_kernel!(
            hypercube_last_arrival_reduce_kernel::<F, I>,
            problem,
            client,
            binding,
            blueprint,
            grid,
            partials.arg(),
            partial_indices.map(Bound::arg).into(),
            arrivals,
            ReduceSplitExtentsLaunch::new(extents.part_count, extents.value_count),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReduceTensors;
    use crate::kernels::testing::{Reduction, arrivals_of, given_workspace, launch_unbound};
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;
    use cubek_std::routine::ROUTINE_LISTS_NO_SETTING;
    use cubek_test_utils::{TestOutcome, ValidationResult};

    prepares_more_row_axes_than_the_grid_holds!(HypercubeLastArrivalReduceRoutine);
    declines_lacking_every_feature!(HypercubeLastArrivalReduceRoutine);
    names_its_setting!(
        HypercubeLastArrivalReduceRoutine,
        "planes=4 lane_slots=32 unit_lines=32"
    );
    estimates_its_microkernel!(
        HypercubeLastArrivalReduceRoutine,
        ReduceRoutineProblem::work_item
    );

    /// `problem` with as much scratch as it asks for.
    fn given_scratch(problem: ReduceRoutineProblem) -> ReduceRoutineProblem {
        ReduceRoutineProblem {
            partials: Some(ReducePartials::UNBOUNDED),
            ..problem
        }
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
        assert_eq!(declined, WRITES_NO_PARTIALS);
    }

    #[test]
    fn declines_partials_shorter_than_its_parts() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem {
            partials: Some(ReducePartials::default()),
            ..ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum)
        };
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let declined = HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings)
            .expect_err("the partials hold no cell");
        assert_eq!(declined, PARTIALS_TOO_SHORT);
    }

    /// The partials a workspace is sized for hold those of a launch at its last setting, its shortest
    /// parts and so its most partials.
    #[test]
    fn the_bound_holds_the_shortest_parts_partials() {
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
        let needs = partials_of(&problem, &blueprint);
        let bound = HypercubeLastArrivalReduceRoutine::partials_bound(&hw, &problem);
        assert!(bound.holds(&needs), "{bound:?} against {needs:?}");
    }

    #[test]
    fn declines_a_row_past_32_bits() {
        let hw = devices::m2_pro();
        let problem = given_scratch(ReduceRoutineProblem::dense(
            &[1, 1 << 32],
            1,
            ReduceOperation::Sum,
        ));
        let settings = HypercubeLastArrivalReduceRoutine::settings(&hw)[0];
        let declined = HypercubeLastArrivalReduceRoutine::prepare(&hw, &problem, &settings)
            .expect_err("32 bits count no such row");
        assert_eq!(declined, ROW_PAST_32_BITS);
    }

    /// The hypercube launched on `tensors` at the last of its settings the device prepares, its
    /// shortest parts, so that short rows still split.
    ///
    /// # Errors
    ///
    /// Why it prepares none for the problem on this device.
    fn launch_shortest_parts(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Decline> {
        let (blueprint, grid) = HypercubeLastArrivalReduceRoutine::settings(hw)
            .iter()
            .rev()
            .map(|settings| {
                HypercubeLastArrivalReduceRoutine::prepare(hw, tensors.problem(), settings)
            })
            .reduce(|first, next| first.or(next))
            .unwrap_or(Err(ROUTINE_LISTS_NO_SETTING))?;
        HypercubeLastArrivalReduceRoutine::launch(tensors, blueprint, grid);
        Ok(())
    }

    /// A launch leaves its arrival counters at zero, so its caller's workspace serves the next
    /// launch: a `[2, 3, 5000]` sum along its contiguous dim, in three parts.
    #[test]
    fn a_launch_leaves_its_counters_at_zero() {
        let client = cubecl::test_device().client();
        let reduction = Reduction {
            shape: vec![2, 3, 5000],
            axis: 2,
            operation: ReduceOperation::Sum,
        };
        let workspace = given_workspace(&client, &reduction);
        let tensors = ReduceTensors::zeros(&reduction.shape, reduction.axis, reduction.operation)
            .with_workspace(workspace.clone());
        let hw = Hardware::from_client(&client);
        if let Err(decline) = launch_shortest_parts(&hw, &tensors) {
            return TestOutcome::Validated(ValidationResult::Skipped(decline.to_string()))
                .enforce();
        }
        assert!(
            arrivals_of(&client, workspace)
                .iter()
                .all(|count| *count == 0)
        );
    }

    /// A `[2, 3, 5000]` tensor along its contiguous dim, in three parts.
    mod along_the_contiguous_dim {
        use super::*;
        every_operation!(launch_shortest_parts, [2, 3, 5000], 2);
    }

    /// A `[3, 2100, 4]` tensor along its middle dim, across the contiguous one, in five parts.
    mod across_the_contiguous_dim {
        use super::*;
        every_operation!(launch_shortest_parts, [3, 2100, 4], 1);
    }

    /// Rows three long, of one part.
    mod rows_of_one_part {
        use super::*;
        every_operation!(launch_shortest_parts, [2, 3, 5000], 1);
    }

    /// Launched unbound, as a fused read stands in for, along the contiguous dim.
    mod unbound {
        use super::*;
        unbound_operations!(
            launch_unbound::<HypercubeLastArrivalReduceRoutine, _>,
            [2, 3, 5000],
            2
        );
    }

    reduces_more_row_axes_than_the_grid_has!(launch_shortest_parts);
    reduces_with_indices!(launch_shortest_parts);
}
