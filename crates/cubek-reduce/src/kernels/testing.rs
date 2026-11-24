//! What the reduce kernels' tests share: a reduction, its operands, the scopes that launch it and
//! check it against the reference, the macros that state a test for each operation, and the
//! problems and devices a host test sweeps.

use crate::eval::EvalWorkspace;
use crate::eval::cpu_reference::ReduceReference;
use crate::kernels::{
    ReduceElements, ReducePartials, ReduceRoutine, ReduceRoutineProblem, ReduceTensors,
    ReduceWorkspace,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Decline, StoredTileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubek_microkernel::reduce::{ReduceOperation, ReduceWrites};
use cubek_std::{Hardware, devices};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};
use std::sync::Arc;

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
    const UNWRITTEN: f32 = -7.0;
    let shape = reduction.out_shape();
    let cells = shape.iter().product();
    tensor_of(
        client,
        &shape,
        reduction.out_element(),
        vec![UNWRITTEN; cells],
    )
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
    R::launch_unbound::<StoredTileOperand>(&tensors, blueprint, grid, bound_back(input));
    Ok(())
}

/// Launches a reduction as `launch` does, and checks it against the reference, row by row.
struct ReductionScopeUnderTest<L> {
    launch: L,
}

impl<L> ScopeUnderTest for ReductionScopeUnderTest<L>
where
    L: Fn(&Client, TensorBinding, TensorBinding, &Reduction) -> Result<(), Decline>,
{
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let input = tensor_of(client, &reduction.shape, F32, reduction.input()).binding();
        let out = unwritten_output(client, reduction);
        (self.launch)(client, input, out.clone().binding(), reduction)?;
        Ok(vec![written_output(out, reduction)])
    }

    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let input = HostData::row_major(
            HostDataVec::F32(reduction.input()),
            Shape::from(reduction.shape.clone()),
        );
        let reference = ReduceReference::new(&input, reduction.axis, reduction.operation, None);
        vec![reference.output]
    }

    /// Whole numbers sum exactly; a mean divides once.
    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        1e-6
    }
}

/// A sum whose rows every routine serves.
pub(crate) fn sum_of_long_rows() -> Reduction {
    Reduction {
        shape: vec![4, 9000],
        axis: 1,
        operation: ReduceOperation::Sum,
    }
}

/// `reduction` launched once by `launch` and checked against the reference; a launch the device
/// declines is a skip.
pub(crate) fn reduces_like_the_cpu(launch: BufferLaunch, reduction: Reduction) {
    let launch_buffers =
        |client: &Client, input: TensorBinding, out: TensorBinding, reduction: &Reduction| {
            let tensors = buffer_tensors(client, input, out, reduction);
            launch(&Hardware::from_client(client), &tensors)
        };
    ReductionScopeUnderTest {
        launch: launch_buffers,
    }
    .check([reduction]);
}

/// [`reduces_like_the_cpu`] of an input the caller binds.
pub(crate) fn reduces_unbound_like_the_cpu(launch: UnboundLaunch, reduction: Reduction) {
    ReductionScopeUnderTest { launch }.check([reduction]);
}

/// The devices a host test sweeps: each kind cubek knows, and one that launches as few cubes along
/// each axis of a grid as any does.
pub(crate) fn swept_devices() -> Vec<Hardware> {
    const FEWEST_CUBE_COUNT: u32 = 65_535;
    let mut capped = devices::m2_pro();
    Arc::make_mut(&mut capped.properties)
        .hardware
        .max_cube_count = (FEWEST_CUBE_COUNT, FEWEST_CUBE_COUNT, FEWEST_CUBE_COUNT);
    vec![
        devices::strix_halo(),
        devices::wgpu_no_subgroups(),
        devices::rdna_range(),
        devices::cdna_wave64(),
        devices::m2_pro(),
        devices::h100(),
        devices::rtx_pro_6000(),
        devices::cpu_12(),
        capped,
    ]
}

/// The problems a host test sweeps: matrices and tensors reduced along each axis, from no row to as
/// many values as 32 bits count, dense and strided, by each carrier, given scratch and none.
pub(crate) fn swept_problems() -> Vec<ReduceRoutineProblem> {
    const EXTENTS: [usize; 12] = [
        0,
        1,
        3,
        9,
        300,
        4097,
        65_535,
        70_001,
        (1 << 20) + 1,
        1 << 23,
        (1 << 30) + 1,
        u32::MAX as usize,
    ];
    let shapes = EXTENTS
        .iter()
        .flat_map(|&outer| EXTENTS.map(|inner| [outer, inner]))
        .flat_map(|[outer, inner]| [1, 9].map(|middle| [outer, middle, inner]));
    shapes
        .flat_map(|shape| (0..shape.len()).map(move |axis| (shape, axis)))
        .flat_map(|(shape, axis)| SWEPT_OPERATIONS.map(|operation| (shape, axis, operation)))
        .filter(|(shape, axis, operation)| {
            let values: u128 = shape.iter().map(|extent| *extent as u128).product();
            values <= u32::MAX as u128 && shape[*axis] >= operation.written_per_row()
        })
        .flat_map(|(shape, axis, operation)| {
            [F32, ElemType::Float(FloatKind::F16)].map(|input| (shape, axis, operation, input))
        })
        .flat_map(|(shape, axis, operation, input)| {
            [1, 2].map(|step| swept_problem(shape, axis, operation, input, step))
        })
        .flat_map(|problem| {
            [None, Some(ReducePartials::UNBOUNDED)].map(|partials| ReduceRoutineProblem {
                partials,
                ..problem.clone()
            })
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

/// A row-major `shape` of `input`, whose innermost axis steps `step` elements, reduced along `axis`
/// by `operation`.
fn swept_problem(
    shape: [usize; 3],
    axis: usize,
    operation: ReduceOperation,
    input: ElemType,
    step: usize,
) -> ReduceRoutineProblem {
    let [outer, middle, inner] = shape;
    let mut out_shape = shape;
    out_shape[axis] = operation.written_per_row();
    let index = ElemType::UInt(UIntKind::U32);
    ReduceRoutineProblem::stated(
        &Geometry::new(&[
            (outer, middle * inner * step),
            (middle, inner * step),
            (inner, step),
        ]),
        &Geometry::row_major(&out_shape),
        axis,
        operation,
        ReduceElements::for_operation(operation, input, Some(index)),
    )
}

/// A test for each `name: operation`, each one launch by `$launch` of `$shape` reduced along
/// `$axis`, checked by `$check`.
macro_rules! reduction_tests {
    (
        $check:path,
        $launch:expr,
        $shape:expr,
        $axis:expr,
        [$($name:ident: $operation:expr),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                use cubek_microkernel::reduce::ReduceOperation;
                $check($launch, $crate::kernels::testing::Reduction {
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
    ($launch:expr, $shape:expr, $axis:expr) => {
        every_operation!($launch, $shape, $axis, rows_shorter_than_three);
        reduction_tests!($crate::kernels::testing::reduces_like_the_cpu, $launch, $shape, $axis, [
            top_3: ReduceOperation::TopK { k: 3 },
            arg_top_3: ReduceOperation::ArgTopK { k: 3 },
        ]);
    };
    ($launch:expr, $shape:expr, $axis:expr, rows_shorter_than_three) => {
        reduction_tests!($crate::kernels::testing::reduces_like_the_cpu, $launch, $shape, $axis, [
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
            $crate::kernels::testing::reduces_unbound_like_the_cpu,
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
                $crate::kernels::testing::reduces_like_the_cpu,
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
                $crate::kernels::testing::reduces_like_the_cpu,
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
        reduction_tests!($crate::kernels::testing::reduces_like_the_cpu, $launch, $shape, $axis, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            arg_max: ReduceOperation::ArgMax,
            top_3: ReduceOperation::TopK { k: 3 },
        ]);
    };
}
