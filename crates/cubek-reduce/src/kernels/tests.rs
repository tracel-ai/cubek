//! The reduce routines on the tile stack, against a CPU reference: a rank-3 tensor reduced along
//! each of its dims, every operation, its values and its rows as they lie; the hypercube's rows
//! split into parts; and the kernel the rule and the autotune pick.

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceElements, ReduceKernel, ReduceKernelHeuristic, ReduceLauncher,
    ReduceOpStrategy, ReduceTensors, UnitDirectReduceRoutine, launch_kernel, launch_tuned,
    launch_zeroed_atomic_unbound, select,
};
use cubecl::client::Client;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, FloatKind, IntKind, Type};
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Decline, StoredTileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubek_microkernel::reduce::{ReduceOperation, ReduceWrites};
use cubek_std::Hardware;
use cubek_std::routine::{ROUTINE_LISTS_NO_SETTING, Routine};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

const OPERATIONS: [ReduceOperation; 14] = [
    ReduceOperation::Sum,
    ReduceOperation::Prod,
    ReduceOperation::Mean,
    ReduceOperation::Max,
    ReduceOperation::Min,
    ReduceOperation::MaxAbs,
    ReduceOperation::Any,
    ReduceOperation::All,
    ReduceOperation::ArgMax,
    ReduceOperation::ArgMin,
    ReduceOperation::TopK { k: 3 },
    ReduceOperation::ArgTopK { k: 3 },
    ReduceOperation::TopK { k: 1 },
    ReduceOperation::ArgTopK { k: 1 },
];

/// The `k` largest values of `row` and their positions, best first.
fn largest(row: &[f32], k: usize) -> Vec<(f32, usize)> {
    ranked(row, true, k)
}

/// The `k` smallest values of `row` and their positions, best first.
fn smallest(row: &[f32], k: usize) -> Vec<(f32, usize)> {
    ranked(row, false, k)
}

/// The `k` best of `row`, best first: the larger (smaller) value, then the lower index, a NaN
/// ahead of every number.
fn ranked(row: &[f32], largest: bool, k: usize) -> Vec<(f32, usize)> {
    let mut order: Vec<(f32, usize)> = row.iter().copied().zip(0..).collect();
    order.sort_by(|a, b| {
        let by_value = match (a.0.is_nan(), b.0.is_nan()) {
            (true, true) => core::cmp::Ordering::Equal,
            (true, false) => core::cmp::Ordering::Less,
            (false, true) => core::cmp::Ordering::Greater,
            _ if largest => b.0.partial_cmp(&a.0).unwrap(),
            _ => a.0.partial_cmp(&b.0).unwrap(),
        };
        by_value.then(a.1.cmp(&b.1))
    });
    order.truncate(k);
    order
}

/// What `operation` writes for `row`.
fn reference(row: &[f32], operation: ReduceOperation) -> Vec<f32> {
    let nan = row.iter().any(|v| v.is_nan());
    let keeping_nan = |v: f32| if nan { f32::NAN } else { v };
    match operation {
        ReduceOperation::Sum => vec![row.iter().sum()],
        ReduceOperation::Prod => vec![row.iter().product()],
        ReduceOperation::Mean => vec![row.iter().sum::<f32>() / row.len() as f32],
        ReduceOperation::Max => vec![keeping_nan(
            row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )],
        ReduceOperation::Min => vec![keeping_nan(
            row.iter().copied().fold(f32::INFINITY, f32::min),
        )],
        ReduceOperation::MaxAbs => vec![keeping_nan(
            row.iter()
                .map(|v| v.abs())
                .fold(f32::NEG_INFINITY, f32::max),
        )],
        ReduceOperation::Any => vec![if row.iter().any(|v| *v != 0.0) {
            1.0
        } else {
            0.0
        }],
        ReduceOperation::All => vec![if row.iter().all(|v| *v != 0.0) {
            1.0
        } else {
            0.0
        }],
        ReduceOperation::ArgMax => vec![largest(row, 1)[0].1 as f32],
        ReduceOperation::ArgMin => vec![smallest(row, 1)[0].1 as f32],
        ReduceOperation::TopK { k } => largest(row, k).iter().map(|e| e.0).collect(),
        ReduceOperation::ArgTopK { k } => largest(row, k).iter().map(|e| e.1 as f32).collect(),
    }
}

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len() - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Every row of `values`, of `shape`, along dim `axis`: its values, and the index of its first.
fn rows_along(values: &[f32], shape: &[usize], axis: usize) -> Vec<Vec<f32>> {
    let strides = strides(shape);
    let count: usize = shape.iter().product::<usize>() / shape[axis];
    (0..count)
        .map(|row| {
            // The row's coordinates on every dim but `axis`, outermost first.
            let mut rest = row;
            let mut base = 0;
            for d in (0..shape.len()).rev().filter(|d| *d != axis) {
                base += (rest % shape[d]) * strides[d];
                rest /= shape[d];
            }
            (0..shape[axis])
                .map(|i| values[base + i * strides[axis]])
                .collect()
        })
        .collect()
}

/// Small whole numbers, so every sum is exact; some ties and zeros, and a NaN rare enough that
/// most rows, even long ones, hold none and check their numbers; for a product,
/// `±1`, `2` and `0.5` on a cycle of seven, prime to every extent here so no row repeats one
/// value, with as many `2` as `0.5` so that a long row's product stays exact and finite.
fn values(len: usize, product: bool) -> Vec<f32> {
    (0..len)
        .map(|i| match (product, i % 97) {
            (true, _) => [1.0, -1.0, 2.0, 0.5, 1.0, 2.0, 0.5][i % 7],
            (false, _) if i % 9973 == 13 => f32::NAN,
            (false, 29 | 30) => 9.0,
            (false, 50) => 0.0,
            _ => ((i * 7) % 11) as f32 - 5.0,
        })
        .collect()
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

impl TestedRoutine {
    /// `R` at the setting this choice picks, launched on `tensors`.
    ///
    /// # Errors
    ///
    /// Why `R` prepares no setting for the problem on this device, which the check reports as a
    /// skip.
    fn launch_with<R: ReduceLauncher<M>, M>(
        self,
        hw: &Hardware,
        tensors: &ReduceTensors,
    ) -> Result<(), Decline> {
        let problem = tensors.problem();
        let mut settings = R::settings(hw);
        if let TestedRoutine::HypercubeLastArrival = self {
            settings.reverse();
        }
        let (blueprint, grid) = settings
            .iter()
            .map(|settings| R::prepare(hw, problem, settings))
            .reduce(|first, next| first.or(next))
            .unwrap_or(Err(ROUTINE_LISTS_NO_SETTING))?;
        R::launch(tensors, blueprint, grid);
        Ok(())
    }
}

/// `shape` reduced along `axis` by `operation`, through the routine `routine` names.
#[derive(Debug, Clone)]
struct Reduction {
    routine: TestedRoutine,
    /// The input's extent along each axis, read by axis (Position Usage).
    shape: Vec<usize>,
    axis: usize,
    operation: ReduceOperation,
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

    fn out_shape(&self) -> Vec<usize> {
        let mut out_shape = self.shape.clone();
        out_shape[self.axis] = self.operation.written_per_row();
        out_shape
    }
}

/// A reduce routine of the tile stack against the host, row by row.
struct RoutineScopeUnderTest;

impl ScopeUnderTest for RoutineScopeUnderTest {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let hw = Hardware::from_client(client);
        let (out_elem, out_shape) = (reduction.out_elem(), reduction.out_shape());
        let input = TestInput::builder(client.clone(), Shape::from(reduction.shape.clone()))
            .dtype(F32)
            .custom(reduction.input())
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), Shape::from(out_shape.clone()))
            .dtype(out_elem)
            .custom(vec![-7.0; out_shape.iter().product()])
            .generate_without_host_data();
        let tensors = ReduceTensors::new(
            client,
            input.binding(),
            out.clone().binding(),
            reduction.axis,
            reduction.operation,
            ReduceElements {
                input: F32,
                accumulator: F32,
                out: out_elem,
            },
        );
        let routine = reduction.routine;
        match routine {
            TestedRoutine::UnitDirect => {
                routine.launch_with::<UnitDirectReduceRoutine, _>(&hw, &tensors)?
            }
            TestedRoutine::PlaneShuffle => {
                routine.launch_with::<PlaneShuffleReduceRoutine, _>(&hw, &tensors)?
            }
            TestedRoutine::CubeStaged => {
                routine.launch_with::<CubeStagedReduceRoutine, _>(&hw, &tensors)?
            }
            TestedRoutine::HypercubeLastArrival => {
                routine.launch_with::<HypercubeLastArrivalReduceRoutine, _>(&hw, &tensors)?
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
        let host = match out_elem {
            I32 => HostDataType::I32,
            _ => HostDataType::F32,
        };
        Ok(vec![Written {
            name: "output",
            tensor: out,
            host,
        }])
    }

    /// Every row's reference written where the row's cells lie in the output.
    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let out_shape = reduction.out_shape();
        let out_strides = strides(&out_shape);
        let axis = reduction.axis;
        let mut out = vec![0.0f32; out_shape.iter().product()];
        let rows = rows_along(&reduction.input(), &reduction.shape, axis);
        for (row, values) in rows.iter().enumerate() {
            let mut rest = row;
            let mut base = 0;
            for d in (0..out_shape.len()).rev().filter(|d| *d != axis) {
                base += (rest % out_shape[d]) * out_strides[d];
                rest /= out_shape[d];
            }
            for (cell, want) in reference(values, reduction.operation)
                .into_iter()
                .enumerate()
            {
                out[base + cell * out_strides[axis]] = want;
            }
        }
        vec![HostData::row_major(
            HostDataVec::F32(out),
            Shape::from(out_shape),
        )]
    }

    /// Whole numbers sum exactly; a mean divides once.
    fn tolerance(&self, _reduction: &Reduction) -> f32 {
        1e-6
    }
}

/// Every routine reduces a `[2, 5, 300]` tensor along each dim, every operation whose rows hold
/// what it writes: along the contiguous dim, and across it, each lane its own row. The plane and the
/// cube skip where the device commits to no plane width.
#[test]
fn every_routine_reduces_every_dim_like_the_cpu() {
    let routines = [
        TestedRoutine::UnitDirect,
        TestedRoutine::PlaneShuffle,
        TestedRoutine::CubeStaged,
    ];
    let mut reductions = Vec::new();
    for routine in routines {
        for axis in [2, 1, 0] {
            for operation in OPERATIONS {
                // A row shorter than `k` leaves cells no reference states: what the kernel pads
                // them with is its own.
                if operation.written_per_row() > [2, 5, 300][axis] {
                    continue;
                }
                reductions.push(Reduction {
                    routine,
                    shape: vec![2, 5, 300],
                    axis,
                    operation,
                });
            }
        }
    }
    RoutineScopeUnderTest.check(reductions);
}

/// A reduce routine of the tile stack launched unbound, its input bound by the caller as a plain
/// buffer once the launch is prepared, at the width the routine picks for it: what a fused read
/// stands in for, against the host.
struct UnboundRoutineScopeUnderTest;

/// The tensors `R` reduces unbound: `input`'s geometry, stated at the width the routine picks
/// for it bound, and `out`.
fn unbound_tensors<R: ReduceLauncher<M>, M>(
    client: &Client,
    input: &TensorBinding,
    out: TensorBinding,
    reduction: &Reduction,
) -> Result<ReduceTensors<Geometry>, Decline> {
    let elements = ReduceElements {
        input: F32,
        accumulator: F32,
        out: reduction.out_elem(),
    };
    let (axis, operation) = (reduction.axis, reduction.operation);
    let buffers = ReduceTensors::new(
        client,
        input.clone(),
        out.clone(),
        axis,
        operation,
        elements,
    );
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
    ))
}

/// `input` bound back as the plain buffer it is, at the spec and width the launch derived.
fn bound_back(input: TensorBinding) -> impl FnOnce(Unbound) -> TileTensorLaunch {
    move |bound| TileTensorLaunch::new(input.into_tensor_arg(), F32, bound.vector_size, bound.spec)
}

/// `reduction`'s routine launched unbound over `input`, bound back as the buffer it is, into
/// `out`.
fn launch_unbound<R: ReduceLauncher<M>, M>(
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
        let (out_elem, out_shape) = (reduction.out_elem(), reduction.out_shape());
        let input = TestInput::builder(client.clone(), Shape::from(reduction.shape.clone()))
            .dtype(F32)
            .custom(reduction.input())
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), Shape::from(out_shape.clone()))
            .dtype(out_elem)
            .custom(vec![-7.0; out_shape.iter().product()])
            .generate_without_host_data();
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
        let host = match out_elem {
            I32 => HostDataType::I32,
            _ => HostDataType::F32,
        };
        Ok(vec![Written {
            name: "output",
            tensor: out,
            host,
        }])
    }

    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        RoutineScopeUnderTest.reference(reduction)
    }

    fn tolerance(&self, reduction: &Reduction) -> f32 {
        RoutineScopeUnderTest.tolerance(reduction)
    }
}

/// The unit, plane and cube routines launched unbound reduce a `[2, 5, 300]` tensor along its
/// contiguous dim and across it as they do bound: a sum, a max, a mean and an argmax. The plane
/// and the cube skip where the device commits to no plane width.
#[test]
fn unbound_routines_reduce_like_the_cpu() {
    let routines = [
        TestedRoutine::UnitDirect,
        TestedRoutine::PlaneShuffle,
        TestedRoutine::CubeStaged,
    ];
    let mut reductions = Vec::new();
    for routine in routines {
        for axis in [2, 1] {
            for operation in [
                ReduceOperation::Sum,
                ReduceOperation::Max,
                ReduceOperation::Mean,
                ReduceOperation::ArgMax,
            ] {
                reductions.push(Reduction {
                    routine,
                    shape: vec![2, 5, 300],
                    axis,
                    operation,
                });
            }
        }
    }
    UnboundRoutineScopeUnderTest.check(reductions);
}

/// The sums the atomic hypercube is checked on: one long row; rows along the contiguous dim; rows
/// across it; and four rows of a matrix, whose output's rows lie side by side in memory.
fn atomic_sums() -> Vec<Reduction> {
    let shapes: [(&[usize], usize); 4] = [
        (&[1, 1, 9000], 2),
        (&[2, 3, 5000], 2),
        (&[3, 2100, 4], 1),
        (&[4, 9000], 1),
    ];
    shapes
        .into_iter()
        .map(|(shape, axis)| Reduction {
            routine: TestedRoutine::Picked(ReduceOpStrategy::Specific(
                ReduceKernel::HypercubeAtomic,
            )),
            shape: shape.to_vec(),
            axis,
            operation: ReduceOperation::Sum,
        })
        .collect()
}

/// The atomic hypercube sums rows split into parts, each part added into the zeroed output, on
/// [`atomic_sums`]. It must launch wherever the device adds floats atomically; a sum is the only
/// operation it serves.
#[test]
fn the_atomic_hypercube_sums_split_rows_like_the_cpu() {
    match adds_floats_atomically() {
        true => RoutineScopeUnderTest.check_some(atomic_sums()),
        false => RoutineScopeUnderTest.check(atomic_sums()),
    }
}

/// The atomic hypercube launched unbound sums as it does bound, into an output that starts away
/// from zero, so the zeroing before it is checked too, on [`atomic_sums`].
#[test]
fn the_atomic_hypercube_sums_an_unbound_input_like_the_cpu() {
    match adds_floats_atomically() {
        true => UnboundRoutineScopeUnderTest.check_some(atomic_sums()),
        false => UnboundRoutineScopeUnderTest.check(atomic_sums()),
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

/// The hypercube splits rows into parts, the last one short, and merges them: a `[2, 3, 5000]`
/// tensor along its contiguous dim, in three parts; across it, a `[3, 2100, 4]` tensor along its
/// middle dim, in five; and a row of one part. It skips where the device commits to no plane width
/// or one cube's stores do not reach another within a launch.
#[test]
fn the_hypercube_reduces_split_rows_like_the_cpu() {
    let mut reductions = Vec::new();
    for (shape, axis) in [([2, 3, 5000], 2), ([3, 2100, 4], 1), ([2, 3, 5000], 1)] {
        for operation in OPERATIONS {
            reductions.push(Reduction {
                routine: TestedRoutine::HypercubeLastArrival,
                shape: shape.to_vec(),
                axis,
                operation,
            });
        }
    }
    RoutineScopeUnderTest.check(reductions);
}

/// Every strategy reduces like the CPU wherever it sends a problem: many short rows, one long row,
/// and rows across the contiguous dim, at extents that are not powers of
/// two. A kernel the device does not run, or that declines the problem, is skipped.
#[test]
fn every_strategy_reduces_like_the_cpu() {
    let mut reductions = Vec::new();
    // The pins first, so a failure names the kernel before a policy that picked it.
    let mut strategies = ReduceOpStrategy::all();
    strategies.rotate_left(2);
    for strategy in strategies {
        for (shape, axis) in [([96, 7, 300], 2), ([1, 1, 9000], 2), ([3, 2100, 5], 1)] {
            for operation in [
                ReduceOperation::Sum,
                ReduceOperation::Mean,
                ReduceOperation::ArgMax,
                ReduceOperation::TopK { k: 3 },
            ] {
                reductions.push(Reduction {
                    routine: TestedRoutine::Picked(strategy),
                    shape: shape.to_vec(),
                    axis,
                    operation,
                });
            }
        }
    }
    RoutineScopeUnderTest.check(reductions);
}

/// A buffer of `shape` in f32 on the test device, holding zeros.
fn zeros(client: &Client, shape: &[usize]) -> TensorBinding {
    TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(F32)
        .zeros()
        .generate_without_host_data()
        .binding()
}

/// `ReduceTensors::with_indices` for `operation` along the last axis of a `[4, 8]` input, the
/// values one wide and the indices `indices_shape`.
fn tensors_with_indices(operation: ReduceOperation, indices_shape: &[usize]) -> ReduceTensors {
    let client = cubecl::test_device().client();
    ReduceTensors::with_indices(
        &client,
        zeros(&client, &[4, 8]),
        zeros(&client, &[4, 1]),
        zeros(&client, indices_shape),
        1,
        operation,
        ReduceElements {
            input: F32,
            accumulator: F32,
            out: F32,
        },
        I32,
    )
}

/// A sum's values carry no index: with_indices' documented panic.
#[test]
#[should_panic]
fn indices_beside_a_sum_panic() {
    tensors_with_indices(ReduceOperation::Sum, &[4, 1]);
}

/// Indices of another shape than the values: with_indices' documented panic.
#[test]
#[should_panic]
fn indices_of_another_shape_panic() {
    tensors_with_indices(ReduceOperation::ArgMax, &[1, 4]);
}
