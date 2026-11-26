//! What the reduce kernels' tests share: a reduction, its operands, the scopes that launch it and
//! check it against the reference, the macros that state a test for each operation, and the
//! problems and devices a host test sweeps.

use crate::eval::EvalWorkspace;
use crate::eval::cpu_reference::ReduceReference;
use crate::kernels::{
    ReduceElements, ReducePartials, ReduceRoutine, ReduceRoutineProblem, ReduceTensors,
    ReduceWorkspace,
};
use cubecl::CubeElement;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Decline, StoredTileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubek_microkernel::reduce::{ReduceOperation, ReduceWrites};
use cubek_std::routine::Routine;
use cubek_std::{Hardware, devices};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeChecker, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// Launches a reduction on buffers, or says why the device runs none.
pub(crate) type BufferLaunch = fn(&Hardware, &ReduceTensors) -> Result<(), Decline>;

/// Launches a reduction of an input its caller binds, into an output, or says why the device runs
/// none.
pub(crate) type UnboundLaunch =
    fn(&Client, TensorBinding, TensorBinding, &Reduction) -> Result<(), Decline>;

/// Small whole numbers, so every sum is exact, with ties and zeros; for a product, `±1`, `2` and
/// `0.5` on a cycle of seven, prime to the step between a row's values here so no row repeats one
/// value, with as many `2` as `0.5` so that a long row's product stays exact and finite. A NaN, an
/// infinity or a row of ties is a behavior of its own, which the microkernels' Component Tests
/// check.
fn values(value_count: usize, product: bool) -> Vec<f32> {
    const FACTORS: [f32; 7] = [1.0, -1.0, 2.0, 0.5, 1.0, 2.0, 0.5];
    (0..value_count)
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

/// `shape` reduced along `axis` by `operation`.
#[derive(Debug, Clone)]
pub(crate) struct Reduction {
    /// The tested input's extent along each axis, read by axis (Position Usage) and walked in their
    /// order (Arrival Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
}

impl Reduction {
    /// What it reduces in: f32 values, written in f32, or their indices in i32.
    pub(crate) fn elements(&self) -> ReduceElements {
        ReduceElements {
            input: F32,
            accumulator: F32,
            out: self.out_element(),
        }
    }

    pub(crate) fn out_shape(&self) -> Vec<usize> {
        let mut out_shape = self.shape.clone();
        out_shape[self.axis] = self.operation.written_per_row();
        out_shape
    }

    fn out_element(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }

    fn input(&self) -> Vec<f32> {
        values(
            self.shape.iter().product(),
            self.operation == ReduceOperation::Prod,
        )
    }
}

/// The workspace `reduction`'s caller gives it on `client`, as large as any routine asks.
pub(crate) fn given_workspace(client: &Client, reduction: &Reduction) -> ReduceWorkspace {
    EvalWorkspace::new(
        client,
        &reduction.shape,
        reduction.axis,
        reduction.operation,
        reduction.elements(),
        false,
    )
    .workspace
}

/// `reduction`'s output on `client`, every cell starting away from zero.
pub(crate) fn unwritten_output(client: &Client, reduction: &Reduction) -> TensorHandle {
    tensor_of(
        client,
        &reduction.out_shape(),
        reduction.out_element(),
        unwritten_cells(reduction),
    )
}

/// What each cell of `reduction`'s output holds before its launch.
fn unwritten_cells(reduction: &Reduction) -> Vec<f32> {
    const UNWRITTEN: f32 = -7.0;
    vec![UNWRITTEN; reduction.out_shape().iter().product()]
}

/// `reduction`'s input on `client`.
fn input_buffer(client: &Client, reduction: &Reduction) -> TensorBinding {
    tensor_of(client, &reduction.shape, F32, reduction.input()).binding()
}

/// `out`, written by `reduction`, as the check reads it back.
pub(crate) fn written_output(out: TensorHandle, reduction: &Reduction) -> Written {
    let host = match reduction.out_element() {
        I32 => HostDataType::I32,
        _ => HostDataType::F32,
    };
    Written {
        name: "output",
        tensor: out,
        host,
    }
}

/// What the arrival counters of `workspace`, on `client`, hold.
pub(crate) fn arrivals_of(client: &Client, workspace: ReduceWorkspace) -> Vec<u32> {
    let counters = client
        .read_one(workspace.arrivals)
        .expect("the counters read back");
    u32::from_bytes(&counters).to_vec()
}

/// A buffer of `shape` in f32 on `client`, holding zeros.
pub(crate) fn zeros(client: &Client, shape: &[usize]) -> TensorBinding {
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

/// `reduction` of `input` into `out`, buffers on `client`, given all the scratch it asks.
fn buffer_tensors(
    client: &Client,
    input: TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> ReduceTensors {
    ReduceTensors::new(
        client,
        input,
        out,
        reduction.axis,
        reduction.operation,
        reduction.elements(),
    )
    .with_workspace(given_workspace(client, reduction))
}

/// The tensors `R` reduces unbound: `input`'s geometry, stated at the lanes the routine picks for
/// it bound, and `out`.
///
/// # Errors
///
/// Why `R` prepares no such reduction here.
pub(crate) fn unbound_tensors<R: ReduceRoutine<M>, M>(
    client: &Client,
    input: &TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> Result<ReduceTensors<Geometry>, Decline> {
    let buffers = buffer_tensors(client, input.clone(), out.clone(), reduction);
    let hw = Hardware::from_client(client);
    let (blueprint, _) = R::prepare_first_served(&hw, buffers.problem())?;
    Ok(ReduceTensors::unbound(
        client,
        Geometry::from(input),
        blueprint.lane_count,
        out,
        reduction.axis,
        reduction.operation,
        reduction.elements(),
    )
    .with_workspace(given_workspace(client, reduction)))
}

/// `input` bound back as the plain buffer it is, at the geometry, spec and lanes the launch
/// derived: its row axes may be merged into one.
pub(crate) fn bound_back(mut input: TensorBinding) -> impl FnOnce(Unbound) -> TileTensorLaunch {
    move |bound| {
        input.shape = bound.geometry.shape().iter().copied().collect();
        input.strides = bound.geometry.strides().iter().copied().collect();
        TileTensorLaunch::new(input.into_tensor_arg(), F32, bound.vector_size, bound.spec)
    }
}

/// `reduction` by `R` launched unbound over `input`, bound back as the buffer it is, into `out`.
///
/// # Errors
///
/// As [`unbound_tensors`].
pub(crate) fn launch_unbound<R: ReduceRoutine<M>, M>(
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

/// Launches a reduction on buffers and checks it against the reference, row by row.
pub(crate) struct BufferScopeChecker {
    launch: BufferLaunch,
}

impl BufferScopeChecker {
    pub fn new(launch: BufferLaunch) -> Self {
        Self { launch }
    }
}

impl ScopeChecker for BufferScopeChecker {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let out = unwritten_output(client, reduction);
        let tensors = buffer_tensors(
            client,
            input_buffer(client, reduction),
            out.clone().binding(),
            reduction,
        );
        (self.launch)(&Hardware::from_client(client), &tensors)?;
        Ok(vec![written_output(out, reduction)])
    }

    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        vec![reference_by_operation(reduction, reduction.operation)]
    }

    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        TOLERANCE
    }
}

/// Launches a reduction whose input the caller binds as a plain buffer once the launch is prepared,
/// what a fused read stands in for, and checks it against the reference.
pub(crate) struct UnboundScopeChecker {
    launch: UnboundLaunch,
}

impl UnboundScopeChecker {
    pub fn new(launch: UnboundLaunch) -> Self {
        Self { launch }
    }
}

impl ScopeChecker for UnboundScopeChecker {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let out = unwritten_output(client, reduction);
        let input = input_buffer(client, reduction);
        (self.launch)(client, input, out.clone().binding(), reduction)?;
        Ok(vec![written_output(out, reduction)])
    }

    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        vec![reference_by_operation(reduction, reduction.operation)]
    }

    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        TOLERANCE
    }
}

/// Launches a reduction on buffers with its values' indices written beside them, and checks both
/// against the reference. The reduction is the one whose indices alone are written.
pub(crate) struct IndicesScopeChecker {
    launch: BufferLaunch,
}

impl IndicesScopeChecker {
    pub fn new(launch: BufferLaunch) -> Self {
        Self { launch }
    }
}

impl ScopeChecker for IndicesScopeChecker {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let indices = unwritten_output(client, reduction);
        let values = tensor_of(
            client,
            &reduction.out_shape(),
            F32,
            unwritten_cells(reduction),
        );
        let elements = ReduceElements {
            out: F32,
            ..reduction.elements()
        };
        let workspace = EvalWorkspace::new(
            client,
            &reduction.shape,
            reduction.axis,
            reduction.operation,
            elements,
            true,
        )
        .workspace;
        let tensors = ReduceTensors::with_indices(
            client,
            input_buffer(client, reduction),
            values.clone().binding(),
            indices.clone().binding(),
            reduction.axis,
            reduction.operation,
            elements,
            I32,
        )
        .with_workspace(workspace);
        (self.launch)(&Hardware::from_client(client), &tensors)?;
        Ok(vec![
            Written {
                name: "values",
                tensor: values,
                host: HostDataType::F32,
            },
            written_output(indices, reduction),
        ])
    }

    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let values = match reduction.operation {
            ReduceOperation::ArgMax => ReduceOperation::Max,
            ReduceOperation::ArgMin => ReduceOperation::Min,
            ReduceOperation::ArgTopK { k } => ReduceOperation::TopK { k },
            operation => unreachable!("{operation:?} writes no indices beside its values"),
        };
        [values, reduction.operation]
            .map(|operation| reference_by_operation(reduction, operation))
            .into()
    }

    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        TOLERANCE
    }
}

/// How far a check lets a launch sit from the reference: whole numbers sum exactly, and a mean
/// divides once.
const TOLERANCE: f32 = 1e-6;

/// What the reduce reference writes for `reduction`'s input, reduced by `operation`.
fn reference_by_operation(reduction: &Reduction, operation: ReduceOperation) -> HostData {
    let input = HostData::row_major(
        HostDataVec::F32(reduction.input()),
        Shape::from(reduction.shape.clone()),
    );
    ReduceReference::new(&input, reduction.axis, operation, None).output
}

/// A sum whose rows every routine serves.
pub(crate) fn sum_of_long_rows() -> Reduction {
    Reduction {
        shape: vec![4, 9000],
        axis: 1,
        operation: ReduceOperation::Sum,
    }
}

/// A sum of one row, which a kernel splits across cubes.
pub(crate) fn sum_of_one_long_row() -> Reduction {
    Reduction {
        shape: vec![1, 70_001],
        axis: 1,
        operation: ReduceOperation::Sum,
    }
}

/// `tensors` given a workspace whose every buffer holds one cell, fewer than any row split across
/// cubes is written to.
pub(crate) fn given_too_little_scratch(tensors: &ReduceTensors) -> ReduceTensors {
    let one_cell = || tensors.client().empty(size_of::<f32>());
    tensors.with_workspace(ReduceWorkspace {
        partials: one_cell(),
        partial_indices: one_cell(),
        arrivals: one_cell(),
    })
}

/// The problems a host test sweeps: matrices reduced along each axis, from no row to as many values
/// as 32 bits count, dense and strided, by each carrier, given scratch.
pub(crate) fn swept_problems() -> Vec<ReduceRoutineProblem> {
    const EXTENT_COUNTS: [usize; 11] = [
        0,
        1,
        3,
        9,
        300,
        4097,
        65_535,
        70_001,
        (1 << 20) + 1,
        (1 << 30) + 1,
        u32::MAX as usize,
    ];
    EXTENT_COUNTS
        .iter()
        .flat_map(|&outer| EXTENT_COUNTS.map(|inner| [outer, inner]))
        .flat_map(|shape| [0, 1].map(|axis| (shape, axis)))
        .flat_map(|(shape, axis)| SWEPT_OPERATIONS.map(|operation| (shape, axis, operation)))
        .filter(|(shape, axis, operation)| {
            let values: u128 = shape.iter().map(|extent| *extent as u128).product();
            values <= u32::MAX as u128 && shape[*axis] >= operation.written_per_row()
        })
        .flat_map(|(shape, axis, operation)| {
            [1, 2].map(|step| swept_problem(shape, axis, operation, step))
        })
        .collect()
}

/// An operation a carrier, the widest top-k one its lane slots hold among them.
const SWEPT_OPERATIONS: [ReduceOperation; 5] = [
    ReduceOperation::Sum,
    ReduceOperation::ArgMax,
    ReduceOperation::TopK { k: 3 },
    ReduceOperation::ArgTopK { k: 20 },
    ReduceOperation::ArgTopK { k: 2048 },
];

/// A row-major `shape`, whose innermost axis steps `step` elements, reduced along `axis` by
/// `operation`, given scratch.
fn swept_problem(
    shape: [usize; 2],
    axis: usize,
    operation: ReduceOperation,
    step: usize,
) -> ReduceRoutineProblem {
    let [outer, inner] = shape;
    let mut out_shape = shape;
    out_shape[axis] = operation.written_per_row();
    let index = ElemType::UInt(UIntKind::U32);
    let stated = ReduceRoutineProblem::stated(
        &Geometry::new(&[(outer, inner * step), (inner, step)]),
        &Geometry::row_major(&out_shape),
        axis,
        operation,
        ReduceElements::for_operation(operation, F32, Some(index)),
    );
    ReduceRoutineProblem {
        partials: Some(ReducePartials::UNBOUNDED),
        ..stated
    }
}

/// A test of `$check` on each device a host test sweeps: each kind cubek knows.
macro_rules! on_each_device {
    ($name:ident, $check:expr) => {
        on_each_device!(@devices $name, $check, [
            strix_halo: cubek_std::devices::strix_halo(),
            wgpu_no_subgroups: cubek_std::devices::wgpu_no_subgroups(),
            rdna_range: cubek_std::devices::rdna_range(),
            cdna_wave64: cubek_std::devices::cdna_wave64(),
            m2_pro: cubek_std::devices::m2_pro(),
            h100: cubek_std::devices::h100(),
            rtx_pro_6000: cubek_std::devices::rtx_pro_6000(),
            cpu_12: cubek_std::devices::cpu_12(),
        ]);
    };
    (@devices $name:ident, $check:expr, [$($device:ident: $hw:expr),* $(,)?]) => {
        mod $name {
            use super::*;

            $(
                #[test]
                fn $device() {
                    $check(&$hw);
                }
            )*
        }
    };
}

/// A test that `$routine` prepares four row axes, more than a launch grid has: every row axis is one
/// [`BATCH`](crate::kernels::BATCH).
macro_rules! prepares_more_row_axes_than_the_grid_holds {
    ($routine:ty) => {
        #[test]
        fn prepares_more_row_axes_than_the_grid_holds() {
            let hw = cubek_std::devices::m2_pro();
            let operation = cubek_microkernel::reduce::ReduceOperation::ArgMax;
            let problem = ReduceRoutineProblem {
                partials: Some($crate::kernels::ReducePartials::UNBOUNDED),
                ..ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, operation)
            };
            let settings = <$routine>::settings(&hw)[0];
            let prepared = <$routine>::prepare(&hw, &problem, &settings).map(|_| ());
            assert!(prepared.is_ok(), "{prepared:?}");
        }
    };
}

/// A test that `$routine`, which reads a Feature, declines a sum on a device lacking every one, as
/// a launch it does not serve, not a broken one.
macro_rules! declines_lacking_every_feature {
    ($routine:ty) => {
        #[test]
        fn declines_lacking_every_feature() {
            let prepared = $crate::kernels::testing::prepared_lacking_every_feature::<$routine>();
            assert!(
                matches!(&prepared, Err(decline) if !decline.is_broken()),
                "{prepared:?}"
            );
        }
    };
}

/// A test that `$routine`, which reads no Feature, prepares a sum on a device lacking every one.
macro_rules! prepares_lacking_every_feature {
    ($routine:ty) => {
        #[test]
        fn prepares_lacking_every_feature() {
            let prepared = $crate::kernels::testing::prepared_lacking_every_feature::<$routine>();
            assert!(prepared.is_ok(), "{prepared:?}");
        }
    };
}

/// Whether `R`'s first setting prepares a sum on a device lacking every Feature.
pub(crate) fn prepared_lacking_every_feature<R: Routine<Problem = ReduceRoutineProblem>>()
-> Result<(), Decline> {
    let hw = devices::lacking_every_feature(devices::m2_pro());
    let problem = ReduceRoutineProblem {
        partials: Some(ReducePartials::UNBOUNDED),
        ..ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum)
    };
    R::prepare(&hw, &problem, &R::settings(&hw)[0]).map(|_| ())
}

/// A test that `$routine`'s first setting is named `$name`, which the tuning cache keys it by.
macro_rules! names_its_setting {
    ($routine:ty, $name:expr) => {
        #[test]
        fn names_its_setting() {
            let setting = <$routine>::settings(&cubek_std::devices::m2_pro())[0];
            assert_eq!(setting.to_string(), $name);
        }
    };
}

/// A test that `$routine` estimates what its microkernel does, handed `$item`.
macro_rules! estimates_its_microkernel {
    ($routine:ty, $item:expr) => {
        #[test]
        fn estimates_its_microkernel() {
            let hw = cubek_std::devices::m2_pro();
            let problem = ReduceRoutineProblem {
                partials: Some($crate::kernels::ReducePartials::UNBOUNDED),
                ..ReduceRoutineProblem::dense(
                    &[4, 4096],
                    1,
                    cubek_microkernel::reduce::ReduceOperation::Sum,
                )
            };
            let (blueprint, grid) = <$routine>::prepare_first_served(&hw, &problem)
                .expect("an M2 Pro runs every routine");
            let mut expected = KernelEstimate::from(&grid);
            blueprint
                .microkernel
                .estimate(&$item(&problem), &(), grid.planes(), &mut expected);
            let mut estimate = KernelEstimate::from(&grid);
            <$routine>::estimate(&problem, &blueprint, &grid, &mut estimate);
            assert_eq!(estimate, expected);
        }
    };
}

/// A test for each `name: operation`, each one launch by `$launch` of `$shape` reduced along
/// `$axis`, checked by `$checker`.
macro_rules! reduction_tests {
    (
        $checker:ident,
        $launch:expr,
        $shape:expr,
        $axis:expr,
        [$($name:ident: $operation:expr),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                use cubek_microkernel::reduce::ReduceOperation;
                use cubek_test_utils::ScopeChecker;
                $crate::kernels::testing::$checker::new($launch).check([
                    $crate::kernels::testing::Reduction {
                        shape: $shape.to_vec(),
                        axis: $axis,
                        operation: $operation,
                    },
                ]);
            }
        )*
    };
}

/// [`reduction_tests`] for every operation whose rows hold what it writes, a top-3 needing three
/// values a row.
macro_rules! every_operation {
    ($launch:expr, $shape:expr, $axis:expr) => {
        every_operation!($launch, $shape, $axis, rows_shorter_than_three);
        reduction_tests!(BufferScopeChecker, $launch, $shape, $axis, [
            top_3: ReduceOperation::TopK { k: 3 },
            arg_top_3: ReduceOperation::ArgTopK { k: 3 },
        ]);
    };
    ($launch:expr, $shape:expr, $axis:expr, rows_shorter_than_three) => {
        reduction_tests!(BufferScopeChecker, $launch, $shape, $axis, [
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

/// [`reduction_tests`] of an input the caller binds, for an operation a kind of carrier.
macro_rules! unbound_operations {
    ($launch:expr, $shape:expr, $axis:expr) => {
        reduction_tests!(
            UnboundScopeChecker,
            $launch,
            $shape,
            $axis,
            [
                sum: ReduceOperation::Sum,
                max: ReduceOperation::Max,
                mean: ReduceOperation::Mean,
                arg_max: ReduceOperation::ArgMax,
            ]
        );
    };
}

/// [`reduction_tests`] that write the indices beside the values: a `[2, 5, 300]` tensor along its
/// contiguous dim and across it, by an operation a carrier that keeps them.
macro_rules! reduces_with_indices {
    ($launch:expr) => {
        /// Launched with the indices written beside the values.
        mod with_indices {
            use super::*;

            mod along_the_contiguous_dim {
                use super::*;
                reduction_tests!(IndicesScopeChecker, $launch, [2, 5, 300], 2, [
                    max: ReduceOperation::ArgMax,
                    top_3: ReduceOperation::ArgTopK { k: 3 },
                ]);
            }

            mod across_the_middle_dim {
                use super::*;
                reduction_tests!(IndicesScopeChecker, $launch, [2, 5, 300], 1, [
                    max: ReduceOperation::ArgMax,
                    top_3: ReduceOperation::ArgTopK { k: 3 },
                ]);
            }
        }
    };
}

/// `$launch` reduces a `[2, 5, 300]` tensor along each dim by every operation, and a
/// `[3, 1001, 36]` one along its middle dim: along the contiguous dim, and across it, each lane
/// its own row. The plane and the cube skip where the device commits to no plane width.
macro_rules! reduces_each_dim {
    ($launch:expr, $unbound:expr) => {
        /// Rows 300 long, along the contiguous dim.
        mod along_the_contiguous_dim {
            use super::*;
            every_operation!($launch, [2, 5, 300], 2);
        }

        /// Rows 5 long, across the contiguous dim.
        mod across_the_middle_dim {
            use super::*;
            every_operation!($launch, [2, 5, 300], 1);
        }

        /// Rows 1001 long across the contiguous dim, 108 of them: whole steps and a last part
        /// step for a plane or a cube whose units lie across lines of rows, and a last box of
        /// rows with fewer lines than columns.
        mod across_a_long_middle_dim {
            use super::*;
            every_operation!($launch, [3, 1001, 36], 1);
        }

        /// Rows 2 long, across the contiguous dim, too short for a top-3.
        mod across_the_outer_dim {
            use super::*;
            every_operation!($launch, [2, 5, 300], 0, rows_shorter_than_three);
        }

        /// Launched unbound, as a fused read stands in for, the input bound back as the plain
        /// buffer it is.
        mod unbound {
            mod along_the_contiguous_dim {
                use super::super::*;
                unbound_operations!($unbound, [2, 5, 300], 2);
            }

            mod across_the_middle_dim {
                use super::super::*;
                unbound_operations!($unbound, [2, 5, 300], 1);
            }
        }
    };
}

/// A tensor with more row axes than a launch grid has, `[2, 3, 2, 2, 64]`, its row axes one
/// [`BATCH`](crate::kernels::BATCH): rows on both sides of the reduced axis, or all before it.
macro_rules! reduces_more_row_axes_than_the_grid_has {
    ($launch:expr) => {
        /// More row axes than a grid has, the rows on both sides of the reduced axis.
        mod more_row_axes_around_the_reduced_one {
            use super::*;
            reduction_tests!(
                BufferScopeChecker,
                $launch,
                [2, 3, 2, 2, 64],
                2,
                [
                    sum: ReduceOperation::Sum,
                    mean: ReduceOperation::Mean,
                    arg_max: ReduceOperation::ArgMax,
                ]
            );
        }

        /// More row axes than a grid has, all before the reduced axis.
        mod more_row_axes_before_the_reduced_one {
            use super::*;
            reduction_tests!(
                BufferScopeChecker,
                $launch,
                [2, 3, 2, 2, 64],
                4,
                [
                    sum: ReduceOperation::Sum,
                    mean: ReduceOperation::Mean,
                    arg_max: ReduceOperation::ArgMax,
                ]
            );
        }
    };
}

/// `$launch` reduces like the CPU wherever it sends a problem: many short rows, one long row, and
/// rows across the contiguous dim, at extents that are not powers of two. A kernel the device does
/// not run, or that declines the problem, is skipped.
macro_rules! strategy_reduces_each_problem {
    ($launch:expr) => {
        /// `[96, 7, 300]` along its contiguous dim: many short rows.
        mod many_short_rows {
            use super::*;
            strategy_operations!($launch, [96, 7, 300], 2);
        }

        /// `[1, 1, 9000]` along its contiguous dim: one long row.
        mod one_long_row {
            use super::*;
            strategy_operations!($launch, [1, 1, 9000], 2);
        }

        /// `[3, 2100, 5]` along its middle dim: rows across the contiguous dim.
        mod rows_across_the_contiguous_dim {
            use super::*;
            strategy_operations!($launch, [3, 2100, 5], 1);
        }
    };
}

/// [`reduction_tests`] for an operation a kind of carrier.
macro_rules! strategy_operations {
    ($launch:expr, $shape:expr, $axis:expr) => {
        reduction_tests!(BufferScopeChecker, $launch, $shape, $axis, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            arg_max: ReduceOperation::ArgMax,
            top_3: ReduceOperation::TopK { k: 3 },
        ]);
    };
}
