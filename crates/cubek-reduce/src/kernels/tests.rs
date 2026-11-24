//! The reduce routines on the tile stack, against a CPU reference: a rank-3 tensor reduced along
//! each of its dims, every operation, its values and its rows as they lie; the hypercube's rows
//! split into parts; and the kernel the rule and the autotune pick.

use crate::eval::cpu_reference;
use crate::eval::workspace::workspace_for;
use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceElements, ReduceKernel, ReduceKernelHeuristic,
    ReduceOpStrategy, ReduceRoutine, ReduceTensors, ReduceWorkspace, UnitDirectReduceRoutine,
    UnitDirectZeroReduceRoutine, launch_kernel, launch_tuned, launch_zeroed_atomic_unbound, select,
};
use cubecl::client::Client;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, FloatKind, IntKind, Type};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Decline, StoredTileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubek_microkernel::reduce::{ReduceOperation, ReduceWrites};
use cubek_std::Hardware;
use cubek_std::routine::{ROUTINE_LISTS_NO_SETTING, Routine};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// Small whole numbers, so every sum is exact, with ties and zeros; for a product, `±1`, `2` and
/// `0.5` on a cycle of seven, prime to the step between a row's values here so no row repeats one
/// value, with as many `2` as `0.5` so that a long row's product stays exact and finite. A NaN, an infinity or a row
/// of ties is a behavior of its own, which the microkernels' Component Tests check.
fn values(len: usize, product: bool) -> Vec<f32> {
    const FACTORS: [f32; 7] = [1.0, -1.0, 2.0, 0.5, 1.0, 2.0, 0.5];
    (0..len)
        .map(|i| match product {
            true => FACTORS[i % FACTORS.len()],
            false => whole_value_at_index(i),
        })
        .collect()
}

/// The value at `index` of a row that is not a product's: a run of ties, a zero, and between them
/// small whole numbers on a cycle of eleven.
fn whole_value_at_index(index: usize) -> f32 {
    match index % 97 {
        29 | 30 => 9.0,
        50 => 0.0,
        _ => ((index * 7) % 11) as f32 - 5.0,
    }
}

/// Which routine a reduction launches, and at which of its settings: the first the device
/// prepares the problem at, or for the hypercube the last, its shortest parts, so that short rows
/// still split.
#[derive(Debug, Clone, Copy)]
enum TestedRoutine {
    UnitDirect,
    PlaneShuffle,
    CubeStaged,
    HypercubeLastArrival,
    /// The kernel a strategy picks: the rule's, the autotune's, or the one it names.
    Picked(ReduceOpStrategy),
}

/// `R` launched on `tensors` at the setting `routine` picks.
///
/// # Errors
///
/// Why `R` prepares no setting for the problem on this device, which the check reports as a
/// skip.
fn launch_tested<R: ReduceRoutine<M>, M>(
    routine: TestedRoutine,
    hw: &Hardware,
    tensors: &ReduceTensors,
) -> Result<(), Decline> {
    let problem = tensors.problem();
    let mut settings = R::settings(hw);
    if let TestedRoutine::HypercubeLastArrival = routine {
        settings.reverse();
    }
    let (blueprint, grid) = settings
        .iter()
        .map(|settings| R::prepare(hw, problem, settings))
        .reduce(|first, next| first.or(next))
        .unwrap_or(Err(ROUTINE_LISTS_NO_SETTING))?;
    R::launch(tensors, blueprint, grid)
}

/// `shape` reduced along `axis` by `operation`, through the routine `routine` names.
#[derive(Debug, Clone)]
struct Reduction {
    pub routine: TestedRoutine,
    /// The tested input's extent along each axis, read by axis (Position Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
}

impl Reduction {
    fn input(&self) -> Vec<f32> {
        values(
            self.shape.iter().product(),
            self.operation == ReduceOperation::Prod,
        )
    }

    fn out_elem(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }

    /// What it reduces in: f32 values, written in the output's element.
    fn elements(&self) -> ReduceElements {
        ReduceElements {
            input: F32,
            accumulator: F32,
            out: self.out_elem(),
        }
    }

    /// The workspace its caller gives it on `client`, as large as any routine asks.
    fn workspace(&self, client: &Client) -> ReduceWorkspace {
        workspace_for(
            client,
            &self.shape,
            self.axis,
            self.operation,
            self.elements(),
            false,
        )
    }

    fn out_shape(&self) -> Vec<usize> {
        let mut out_shape = self.shape.clone();
        out_shape[self.axis] = self.operation.written_per_row();
        out_shape
    }
}

/// `reduction`'s input on `client`.
fn input_tensor(client: &Client, reduction: &Reduction) -> TensorHandle {
    tensor_of(client, &reduction.shape, F32, reduction.input())
}

/// `reduction`'s output on `client`, every cell starting away from zero.
fn unwritten_output(client: &Client, reduction: &Reduction) -> TensorHandle {
    const UNWRITTEN: f32 = -7.0;
    let shape = reduction.out_shape();
    let cells = shape.iter().product();
    tensor_of(client, &shape, reduction.out_elem(), vec![UNWRITTEN; cells])
}

/// `out`, written by `reduction`, as the check reads it back.
fn written_output(out: TensorHandle, reduction: &Reduction) -> Written {
    let host = match reduction.out_elem() {
        I32 => HostDataType::I32,
        _ => HostDataType::F32,
    };
    Written {
        name: "output",
        tensor: out,
        host,
    }
}

/// A buffer of `shape` in f32 on `client`, holding zeros.
pub(super) fn zeros(client: &Client, shape: &[usize]) -> TensorBinding {
    let cells = shape.iter().product();
    tensor_of(client, shape, F32, vec![0.0; cells]).binding()
}

/// A tensor of `shape` in `element` on `client`, holding `values` in row-major order.
fn tensor_of(
    client: &Client,
    shape: &[usize],
    element: ElemType,
    values: Vec<f32>,
) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(element)
        .custom(values)
        .generate_without_host_data()
}

/// A reduce routine of the tile stack against the host, row by row.
struct RoutineScopeUnderTest;

impl ScopeUnderTest for RoutineScopeUnderTest {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let hw = Hardware::from_client(client);
        let input = input_tensor(client, reduction);
        let out = unwritten_output(client, reduction);
        let tensors = ReduceTensors::new(
            client,
            input.binding(),
            out.clone().binding(),
            reduction.axis,
            reduction.operation,
            reduction.elements(),
        )
        .with_workspace(reduction.workspace(client));
        let routine = reduction.routine;
        match routine {
            TestedRoutine::UnitDirect => {
                launch_tested::<UnitDirectReduceRoutine, _>(routine, &hw, &tensors)?
            }
            TestedRoutine::PlaneShuffle => {
                launch_tested::<PlaneShuffleReduceRoutine, _>(routine, &hw, &tensors)?
            }
            TestedRoutine::CubeStaged => {
                launch_tested::<CubeStagedReduceRoutine, _>(routine, &hw, &tensors)?
            }
            TestedRoutine::HypercubeLastArrival => {
                launch_tested::<HypercubeLastArrivalReduceRoutine, _>(routine, &hw, &tensors)?
            }
            TestedRoutine::Picked(ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile)) => {
                select(&hw, &tensors)?
            }
            TestedRoutine::Picked(ReduceOpStrategy::Specific(kernel)) => {
                launch_kernel(kernel, &hw, &tensors)?
            }
            TestedRoutine::Picked(ReduceOpStrategy::Autotune) => {
                launch_tuned(tensors, &cubecl::test_device())
                    .expect("the floor serves every problem")
            }
        }
        Ok(vec![written_output(out, reduction)])
    }

    /// What the reduce reference writes for the input.
    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let input = HostData::row_major(
            HostDataVec::F32(reduction.input()),
            Shape::from(reduction.shape.clone()),
        );
        vec![cpu_reference::reference(
            &input,
            reduction.axis,
            reduction.operation,
            None,
        )]
    }

    /// Whole numbers sum exactly; a mean divides once.
    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        1e-6
    }
}

/// A reduce routine of the tile stack launched unbound, its input bound by the caller as a plain
/// buffer once the launch is prepared, at the width the routine picks for it: what a fused read
/// stands in for, against the host.
struct UnboundRoutineScopeUnderTest;

/// The tensors `R` reduces unbound: `input`'s geometry, stated at the width the routine picks
/// for it bound, and `out`.
fn unbound_tensors<R: ReduceRoutine<M>, M>(
    client: &Client,
    input: &TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> Result<ReduceTensors<Geometry>, Decline> {
    let elements = reduction.elements();
    let (axis, operation) = (reduction.axis, reduction.operation);
    let buffers = ReduceTensors::new(
        client,
        input.clone(),
        out.clone(),
        axis,
        operation,
        elements,
    )
    .with_workspace(reduction.workspace(client));
    let hw = Hardware::from_client(client);
    let (blueprint, _) = R::prepare_first_served(&hw, buffers.problem())?;
    Ok(ReduceTensors::unbound(
        client,
        Geometry::from(input),
        blueprint.width,
        out,
        axis,
        operation,
        elements,
    )
    .with_workspace(reduction.workspace(client)))
}

/// `input` bound back as the plain buffer it is, at the geometry, spec and width the launch
/// derived: its row axes may be merged into one.
pub(super) fn bound_back(mut input: TensorBinding) -> impl FnOnce(Unbound) -> TileTensorLaunch {
    move |bound| {
        input.shape = bound.geometry.shape().iter().copied().collect();
        input.strides = bound.geometry.strides().iter().copied().collect();
        TileTensorLaunch::new(input.into_tensor_arg(), F32, bound.vector_size, bound.spec)
    }
}

/// `reduction`'s routine launched unbound over `input`, bound back as the buffer it is, into
/// `out`.
fn launch_unbound<R: ReduceRoutine<M>, M>(
    client: &Client,
    input: TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> Result<(), Decline> {
    let tensors = unbound_tensors::<R, M>(client, &input, out, reduction)?;
    let hw = Hardware::from_client(client);
    let (blueprint, grid) = R::prepare_first_served(&hw, tensors.problem())?;
    R::launch_unbound::<StoredTileOperand>(&tensors, blueprint, grid, bound_back(input))
}

/// The atomic hypercube launched unbound as [`launch_unbound`] launches a routine, into `out`
/// zeroed first by the zero routine, as its tuned candidate does.
fn launch_atomic_unbound(
    client: &Client,
    input: TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> Result<(), Decline> {
    let tensors =
        unbound_tensors::<HypercubeAtomicReduceRoutine, _>(client, &input, out, reduction)?;
    let hw = Hardware::from_client(client);
    let prepared = HypercubeAtomicReduceRoutine::prepare_first_served(&hw, tensors.problem())?;
    launch_zeroed_atomic_unbound::<StoredTileOperand>(&hw, &tensors, prepared, bound_back(input))
}

impl ScopeUnderTest for UnboundRoutineScopeUnderTest {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let input = input_tensor(client, reduction);
        let out = unwritten_output(client, reduction);
        let (input, written) = (input.binding(), out.clone().binding());
        match reduction.routine {
            TestedRoutine::UnitDirect => {
                launch_unbound::<UnitDirectReduceRoutine, _>(client, input, written, reduction)
            }
            TestedRoutine::PlaneShuffle => {
                launch_unbound::<PlaneShuffleReduceRoutine, _>(client, input, written, reduction)
            }
            TestedRoutine::CubeStaged => {
                launch_unbound::<CubeStagedReduceRoutine, _>(client, input, written, reduction)
            }
            TestedRoutine::HypercubeLastArrival => launch_unbound::<
                HypercubeLastArrivalReduceRoutine,
                _,
            >(client, input, written, reduction),
            TestedRoutine::Picked(ReduceOpStrategy::Specific(ReduceKernel::HypercubeAtomic)) => {
                launch_atomic_unbound(client, input, written, reduction)
            }
            TestedRoutine::Picked(strategy) => {
                unreachable!("an unbound launch names its routine, not a strategy ({strategy:?})")
            }
        }?;
        Ok(vec![written_output(out, reduction)])
    }

    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        RoutineScopeUnderTest.reference(reduction)
    }

    fn tolerance(&self, reduction: &Reduction) -> f32 {
        RoutineScopeUnderTest.tolerance(reduction)
    }
}

/// Whether the test device runs the atomic hypercube's sums: a plane width it commits to, for the
/// cube microkernel each cube runs, and an atomic add of floats. Where it does, a sum must launch;
/// where it does not, the routine's decline is a skip.
fn adds_floats_atomically() -> bool {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    hw.plane_dim().is_ok()
        && hw
            .properties
            .atomic_type_usage(Type::atomic(F32))
            .contains(AtomicUsage::Add)
}

/// `reduction` launched once and checked against the reduce reference; a routine that declines it
/// is a skip.
fn reduces_like_the_cpu(reduction: Reduction) {
    RoutineScopeUnderTest.check([reduction]);
}

/// `reduction` launched once unbound, its input bound back as the plain buffer it is, and checked
/// against the reduce reference; a routine that declines it is a skip.
fn reduces_unbound_like_the_cpu(reduction: Reduction) {
    UnboundRoutineScopeUnderTest.check([reduction]);
}

/// A sum by the atomic hypercube, checked through `scope`: it must launch wherever the device adds
/// floats atomically, and is a skip elsewhere.
fn sums_atomically_like_the_cpu(
    scope: impl ScopeUnderTest<Case = Reduction>,
    reduction: Reduction,
) {
    match adds_floats_atomically() {
        true => scope.check_some([reduction]),
        false => scope.check([reduction]),
    }
}

/// A test for each `name: operation`, each one launch of `check` on `shape` reduced along `axis`
/// by `routine`.
macro_rules! reduction_tests {
    (
        $check:ident,
        $routine:expr,
        $shape:expr,
        $axis:expr,
        [$($name:ident: $operation:expr),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                $check(Reduction {
                    routine: $routine,
                    shape: $shape.to_vec(),
                    axis: $axis,
                    operation: $operation,
                });
            }
        )*
    };
}

/// [`reduction_tests`] for every operation whose rows hold what it writes, a top-3 needing three
/// values a row.
macro_rules! every_operation {
    ($check:ident, $routine:expr, $shape:expr, $axis:expr) => {
        reduction_tests!($check, $routine, $shape, $axis, [
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
        ]);
        reduction_tests!($check, $routine, $shape, $axis, [
            top_3: ReduceOperation::TopK { k: 3 },
            arg_top_3: ReduceOperation::ArgTopK { k: 3 },
        ]);
    };
    ($check:ident, $routine:expr, $shape:expr, $axis:expr, rows_shorter_than_three) => {
        reduction_tests!($check, $routine, $shape, $axis, [
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
        ]);
    };
}

/// Each routine reduces a `[2, 5, 300]` tensor along each dim by every operation, and a
/// `[3, 1001, 36]` one along its middle dim: along the contiguous dim, and across it, each lane
/// its own row. The plane and the cube skip where the device commits to no plane width.
macro_rules! reduces_each_dim {
    ($routine:expr) => {
        /// Rows 300 long, along the contiguous dim.
        mod along_the_contiguous_dim {
            use super::super::*;
            every_operation!(reduces_like_the_cpu, $routine, [2, 5, 300], 2);
        }

        /// Rows 5 long, across the contiguous dim.
        mod across_the_middle_dim {
            use super::super::*;
            every_operation!(reduces_like_the_cpu, $routine, [2, 5, 300], 1);
        }

        /// Rows 1001 long across the contiguous dim, 108 of them: whole steps and a last part
        /// step for a plane or a cube whose units lie across lines of rows, and a last box of
        /// rows with fewer lines than columns.
        mod across_a_long_middle_dim {
            use super::super::*;
            every_operation!(reduces_like_the_cpu, $routine, [3, 1001, 36], 1);
        }

        /// Rows 2 long, across the contiguous dim, too short for a top-3.
        mod across_the_outer_dim {
            use super::super::*;
            every_operation!(reduces_like_the_cpu, $routine, [2, 5, 300], 0, rows_shorter_than_three);
        }

        /// Launched unbound, as a fused read stands in for, the input bound back as the plain
        /// buffer it is.
        mod unbound {
            mod along_the_contiguous_dim {
                use super::super::super::*;
                reduction_tests!(reduces_unbound_like_the_cpu, $routine, [2, 5, 300], 2, [
                    sum: ReduceOperation::Sum,
                    max: ReduceOperation::Max,
                    mean: ReduceOperation::Mean,
                    arg_max: ReduceOperation::ArgMax,
                ]);
            }

            mod across_the_middle_dim {
                use super::super::super::*;
                reduction_tests!(reduces_unbound_like_the_cpu, $routine, [2, 5, 300], 1, [
                    sum: ReduceOperation::Sum,
                    max: ReduceOperation::Max,
                    mean: ReduceOperation::Mean,
                    arg_max: ReduceOperation::ArgMax,
                ]);
            }
        }
    };
}

/// A tensor with more row axes than a launch grid has, `[2, 3, 2, 2, 64]`, each routine's row axes
/// one [`BATCH`](crate::kernels::BATCH): rows on both sides of the reduced axis, or all before it.
macro_rules! reduces_more_row_axes_than_the_grid_has {
    ($routine:expr) => {
        /// More row axes than a grid has, the rows on both sides of the reduced axis.
        mod more_row_axes_around_the_reduced_one {
            use super::super::*;
            reduction_tests!(reduces_like_the_cpu, $routine, [2, 3, 2, 2, 64], 2, [
                sum: ReduceOperation::Sum,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
            ]);
        }

        /// More row axes than a grid has, all before the reduced axis.
        mod more_row_axes_before_the_reduced_one {
            use super::super::*;
            reduction_tests!(reduces_like_the_cpu, $routine, [2, 3, 2, 2, 64], 4, [
                sum: ReduceOperation::Sum,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
            ]);
        }
    };
}

/// The unit routine: a unit reduces each row.
mod unit_direct {
    reduces_each_dim!(TestedRoutine::UnitDirect);
    reduces_more_row_axes_than_the_grid_has!(TestedRoutine::UnitDirect);
}

/// The plane routine: a plane reduces each row, its lanes meeting by shuffles.
mod plane_shuffle {
    reduces_each_dim!(TestedRoutine::PlaneShuffle);
    reduces_more_row_axes_than_the_grid_has!(TestedRoutine::PlaneShuffle);
}

/// The cube routine: a cube reduces each row, its planes meeting in shared memory.
mod cube_staged {
    reduces_each_dim!(TestedRoutine::CubeStaged);
    reduces_more_row_axes_than_the_grid_has!(TestedRoutine::CubeStaged);
}

/// The last-arrival hypercube, at its shortest parts: rows split into parts, the last one short,
/// and merged by the last cube to arrive. It skips where the device commits to no plane width or
/// one cube's stores do not reach another within a launch.
mod hypercube_last_arrival {
    use super::*;
    use cubecl::CubeElement;
    use cubek_test_utils::{TestOutcome, ValidationResult};

    /// A launch leaves its arrival counters at zero, so its caller's workspace serves the next
    /// launch: a `[2, 3, 5000]` sum along its contiguous dim, in three parts.
    #[test]
    fn a_launch_leaves_its_counters_at_zero() {
        let client = cubecl::test_device().client();
        let reduction = Reduction {
            routine: TestedRoutine::HypercubeLastArrival,
            shape: vec![2, 3, 5000],
            axis: 2,
            operation: ReduceOperation::Sum,
        };
        let workspace = reduction.workspace(&client);
        let tensors = ReduceTensors::new(
            &client,
            zeros(&client, &reduction.shape),
            zeros(&client, &reduction.out_shape()),
            reduction.axis,
            reduction.operation,
            reduction.elements(),
        )
        .with_workspace(workspace.clone());
        let hw = Hardware::from_client(&client);
        if let Err(decline) =
            launch_tested::<HypercubeLastArrivalReduceRoutine, _>(reduction.routine, &hw, &tensors)
        {
            return TestOutcome::Validated(ValidationResult::Skipped(decline.to_string()))
                .enforce();
        }
        let counters = client
            .read_one(workspace.arrivals)
            .expect("the counters read back");
        assert!(u32::from_bytes(&counters).iter().all(|count| *count == 0));
    }

    /// A `[2, 3, 5000]` tensor along its contiguous dim, in three parts.
    mod along_the_contiguous_dim {
        use super::super::*;
        every_operation!(
            reduces_like_the_cpu,
            TestedRoutine::HypercubeLastArrival,
            [2, 3, 5000],
            2
        );
    }

    /// A `[3, 2100, 4]` tensor along its middle dim, across the contiguous one, in five parts.
    mod across_the_contiguous_dim {
        use super::super::*;
        every_operation!(
            reduces_like_the_cpu,
            TestedRoutine::HypercubeLastArrival,
            [3, 2100, 4],
            1
        );
    }

    /// Rows three long, of one part.
    mod rows_of_one_part {
        use super::super::*;
        every_operation!(
            reduces_like_the_cpu,
            TestedRoutine::HypercubeLastArrival,
            [2, 3, 5000],
            1
        );
    }

    /// Launched unbound, as a fused read stands in for, along the contiguous dim.
    mod unbound {
        use super::super::*;
        reduction_tests!(
            reduces_unbound_like_the_cpu,
            TestedRoutine::HypercubeLastArrival,
            [2, 3, 5000],
            2,
            [
                sum: ReduceOperation::Sum,
                max: ReduceOperation::Max,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
            ]
        );
    }

    reduces_more_row_axes_than_the_grid_has!(TestedRoutine::HypercubeLastArrival);
}

/// The atomic hypercube sums rows split into parts, each part added into the zeroed output. Bound,
/// and unbound into an output that starts away from zero, so the zeroing before it is checked too.
/// A sum is the only operation it serves.
mod hypercube_atomic {
    use super::*;

    /// The atomic hypercube's kernel, named by a strategy that pins it.
    const ATOMIC: TestedRoutine =
        TestedRoutine::Picked(ReduceOpStrategy::Specific(ReduceKernel::HypercubeAtomic));

    /// A sum of `shape` along `axis` by the atomic hypercube.
    fn sum(shape: &[usize], axis: usize) -> Reduction {
        Reduction {
            routine: ATOMIC,
            shape: shape.to_vec(),
            axis,
            operation: ReduceOperation::Sum,
        }
    }

    #[test]
    fn sums_one_long_row() {
        sums_atomically_like_the_cpu(RoutineScopeUnderTest, sum(&[1, 1, 9000], 2));
    }

    #[test]
    fn sums_rows_along_the_contiguous_dim() {
        sums_atomically_like_the_cpu(RoutineScopeUnderTest, sum(&[2, 3, 5000], 2));
    }

    #[test]
    fn sums_rows_across_the_contiguous_dim() {
        sums_atomically_like_the_cpu(RoutineScopeUnderTest, sum(&[3, 2100, 4], 1));
    }

    /// Four rows of a matrix, whose output's rows lie side by side in memory.
    #[test]
    fn sums_the_rows_of_a_matrix() {
        sums_atomically_like_the_cpu(RoutineScopeUnderTest, sum(&[4, 9000], 1));
    }

    #[test]
    fn sums_one_long_row_unbound() {
        sums_atomically_like_the_cpu(UnboundRoutineScopeUnderTest, sum(&[1, 1, 9000], 2));
    }

    #[test]
    fn sums_rows_along_the_contiguous_dim_unbound() {
        sums_atomically_like_the_cpu(UnboundRoutineScopeUnderTest, sum(&[2, 3, 5000], 2));
    }

    #[test]
    fn sums_rows_across_the_contiguous_dim_unbound() {
        sums_atomically_like_the_cpu(UnboundRoutineScopeUnderTest, sum(&[3, 2100, 4], 1));
    }

    #[test]
    fn sums_the_rows_of_a_matrix_unbound() {
        sums_atomically_like_the_cpu(UnboundRoutineScopeUnderTest, sum(&[4, 9000], 1));
    }
}

/// The zero routine alone: a reduction's output, which starts away from zero, holds only zeros
/// after it. What the atomic hypercube adds its parts into.
struct ZeroScopeUnderTest;

impl ScopeUnderTest for ZeroScopeUnderTest {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let out = unwritten_output(client, reduction);
        let tensors = ReduceTensors::new(
            client,
            zeros(client, &reduction.shape),
            out.clone().binding(),
            reduction.axis,
            reduction.operation,
            reduction.elements(),
        );
        let hw = Hardware::from_client(client);
        let (blueprint, grid) =
            UnitDirectZeroReduceRoutine::prepare_first_served(&hw, tensors.problem())?;
        UnitDirectZeroReduceRoutine::launch(&tensors, blueprint, grid)?;
        Ok(vec![written_output(out, reduction)])
    }

    /// Zeros, the output's shape.
    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let out_shape = reduction.out_shape();
        vec![HostData::row_major(
            HostDataVec::F32(vec![0.0; out_shape.iter().product()]),
            Shape::from(out_shape),
        )]
    }

    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        0.0
    }
}

/// The output of `[5, 300, 7]` summed along its middle dim, 35 cells, all zeroed.
#[test]
fn the_zero_routine_zeroes_every_cell_of_the_output() {
    ZeroScopeUnderTest.check([Reduction {
        routine: TestedRoutine::Picked(ReduceOpStrategy::Specific(ReduceKernel::HypercubeAtomic)),
        shape: vec![5, 300, 7],
        axis: 1,
        operation: ReduceOperation::Sum,
    }]);
}

/// Every strategy reduces like the CPU wherever it sends a problem: many short rows, one long row,
/// and rows across the contiguous dim, at extents that are not powers of two. A kernel the device
/// does not run, or that declines the problem, is skipped.
macro_rules! strategy_reduces_each_problem {
    ($strategy:expr) => {
        /// `[96, 7, 300]` along its contiguous dim: many short rows.
        mod many_short_rows {
            use super::super::*;
            reduction_tests!(reduces_like_the_cpu, TestedRoutine::Picked($strategy), [96, 7, 300], 2, [
                sum: ReduceOperation::Sum,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
                top_3: ReduceOperation::TopK { k: 3 },
            ]);
        }

        /// `[1, 1, 9000]` along its contiguous dim: one long row.
        mod one_long_row {
            use super::super::*;
            reduction_tests!(reduces_like_the_cpu, TestedRoutine::Picked($strategy), [1, 1, 9000], 2, [
                sum: ReduceOperation::Sum,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
                top_3: ReduceOperation::TopK { k: 3 },
            ]);
        }

        /// `[3, 2100, 5]` along its middle dim: rows across the contiguous dim.
        mod rows_across_the_contiguous_dim {
            use super::super::*;
            reduction_tests!(reduces_like_the_cpu, TestedRoutine::Picked($strategy), [3, 2100, 5], 1, [
                sum: ReduceOperation::Sum,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
                top_3: ReduceOperation::TopK { k: 3 },
            ]);
        }
    };
}

/// The kernel each strategy picks, or the one it pins.
mod strategy {
    use super::*;

    /// The autotune's pick.
    mod autotune {
        strategy_reduces_each_problem!(ReduceOpStrategy::Autotune);
    }

    /// The rule's pick.
    mod tile_heuristic {
        strategy_reduces_each_problem!(ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile));
    }

    /// The unit kernel, pinned.
    mod unit {
        strategy_reduces_each_problem!(ReduceOpStrategy::Specific(ReduceKernel::Unit));
    }

    /// The plane kernel, pinned.
    mod plane_shuffle {
        strategy_reduces_each_problem!(ReduceOpStrategy::Specific(ReduceKernel::PlaneShuffle));
    }

    /// The cube kernel, pinned.
    mod cube_staged {
        strategy_reduces_each_problem!(ReduceOpStrategy::Specific(ReduceKernel::CubeStaged));
    }

    /// The last-arrival hypercube, pinned.
    mod hypercube_last_arrival {
        strategy_reduces_each_problem!(ReduceOpStrategy::Specific(
            ReduceKernel::HypercubeLastArrival
        ));
    }
}
