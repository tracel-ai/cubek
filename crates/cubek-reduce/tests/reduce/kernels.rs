//! The reduce routines on the tile stack, against a CPU reference: a rank-3 tensor reduced along
//! each of its dims, every operation, its values and its rows as they lie; and the hypercube's
//! rows split into parts.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::zspace::Shape;
use cubek_microkernel::reduce::{ReduceOperation, ReduceWrites};
use cubek_reduce::kernels::{
    CubeReduceRoutine, HypercubeReduceRoutine, PlaneReduceRoutine, ReduceElements, ReduceRoutine,
    ReduceRoutineProblem, ReduceTensors, UnitReduceRoutine, launch_reduce_routine,
};
use cubek_std::Hardware;
use cubek_std::devices;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

const OPERATIONS: [ReduceOperation; 12] = [
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
];

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
        ReduceOperation::ArgMax => vec![ranked(row, true, 1)[0].1 as f32],
        ReduceOperation::ArgMin => vec![ranked(row, false, 1)[0].1 as f32],
        ReduceOperation::TopK { k } => ranked(row, true, k).iter().map(|e| e.0).collect(),
        ReduceOperation::ArgTopK { k } => ranked(row, true, k).iter().map(|e| e.1 as f32).collect(),
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
enum Scope {
    Unit,
    Plane,
    Cube,
    Hypercube,
}

impl Scope {
    /// `routine` at the setting this scope picks, launched on `tensors`.
    ///
    /// # Panics
    ///
    /// Where the routine prepares no setting for the problem, or the launch fails.
    fn launch_with<R: ReduceRoutine<M> + core::fmt::Debug, M>(
        self,
        routine: &R,
        hw: &Hardware,
        tensors: &ReduceTensors,
    ) {
        let problem = tensors.problem();
        let mut settings = routine.settings(hw);
        if let Scope::Hypercube = self {
            settings.reverse();
        }
        let settings = settings
            .into_iter()
            .find(|settings| routine.prepare(hw, problem, settings).is_ok())
            .unwrap_or_else(|| panic!("{routine:?} prepares no setting for {problem:?}"));
        launch_reduce_routine(routine, tensors, &settings)
            .unwrap_or_else(|why| panic!("{routine:?} {problem:?}: {why:?}"));
    }
}

/// `shape` reduced along `axis` by `operation`, through the routine `scope` names.
#[derive(Debug, Clone)]
struct Reduction {
    scope: Scope,
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
struct Routines;

impl ScopeUnderTest for Routines {
    type Case = Reduction;

    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Refusal> {
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
        let scope = reduction.scope;
        match scope {
            Scope::Unit => scope.launch_with(&UnitReduceRoutine, &hw, &tensors),
            Scope::Plane => scope.launch_with(&PlaneReduceRoutine, &hw, &tensors),
            Scope::Cube => scope.launch_with(&CubeReduceRoutine, &hw, &tensors),
            Scope::Hypercube => scope.launch_with(&HypercubeReduceRoutine, &hw, &tensors),
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
/// what it writes: along the contiguous dim, and across it, each lane its own row. The plane and the cube where the device
/// commits to a plane width.
#[test]
fn every_routine_reduces_every_dim_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let mut scopes = vec![Scope::Unit];
    if hw.plane_dim().is_ok() {
        scopes.extend([Scope::Plane, Scope::Cube]);
    }
    let mut reductions = Vec::new();
    for scope in scopes {
        for axis in [2, 1, 0] {
            for operation in OPERATIONS {
                // A row shorter than `k` leaves cells no reference states: what the kernel pads
                // them with is its own.
                if operation.written_per_row() > [2, 5, 300][axis] {
                    continue;
                }
                reductions.push(Reduction {
                    scope,
                    shape: vec![2, 5, 300],
                    axis,
                    operation,
                });
            }
        }
    }
    Routines.check(reductions);
}

/// The hypercube splits rows into parts, the last one short, and merges them: a `[2, 3, 5000]`
/// tensor along its contiguous dim, in three parts; across it, a `[3, 2100, 4]` tensor along its
/// middle dim, in five; and a row of one part. Where the device commits to a plane width and one
/// cube's stores reach another within a launch.
#[test]
fn the_hypercube_reduces_split_rows_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    if hw.plane_dim().is_err() || !hw.properties.features.device_memory_scope {
        return;
    }
    let mut reductions = Vec::new();
    for (shape, axis) in [([2, 3, 5000], 2), ([3, 2100, 4], 1), ([2, 3, 5000], 1)] {
        for operation in OPERATIONS {
            reductions.push(Reduction {
                scope: Scope::Hypercube,
                shape: shape.to_vec(),
                axis,
                operation,
            });
        }
    }
    Routines.check(reductions);
}

/// The problem a host states for `shape` reduced along `axis`, its rows contiguous.
fn problem(shape: &[usize], axis: usize) -> ReduceRoutineProblem {
    let inner = shape.len() - 1;
    ReduceRoutineProblem {
        shape: shape.to_vec(),
        axis,
        operation: ReduceOperation::ArgMax,
        elements: ReduceElements {
            input: F32,
            accumulator: F32,
            out: I32,
        },
        contiguous: Some(inner),
        elem_alignments: [ElemAlignment::new(shape, &strides(shape)); 2],
        indices: None,
    }
}

/// One kernel serves every shape of a rank, layout and width: the rows and the row length are the
/// launch's, so a sequence that grows by a token recompiles nothing, along it or across it.
#[test]
fn every_shape_of_a_layout_compiles_one_kernel() {
    let hw = devices::m2_pro();
    let compiled = |shape: &[usize], axis: usize| {
        let problem = problem(shape, axis);
        let settings = UnitReduceRoutine.settings(&hw)[0];
        let (blueprint, _) = UnitReduceRoutine
            .prepare(&hw, &problem, &settings)
            .expect("the unit routine runs everywhere");
        (
            blueprint.microkernel,
            blueprint.width,
            problem.kernel(&blueprint.partitioning),
        )
    };
    assert_eq!(compiled(&[2, 5, 300], 2), compiled(&[3, 7, 308], 2));
    assert_eq!(compiled(&[2, 5, 300], 1), compiled(&[4, 9, 308], 1));
}

/// Where a launch grid has too few dims for the rows, the cube and the hypercube decline rather
/// than panic, and the unit and the plane still run: a rank-5 tensor, four row dims.
#[test]
fn routines_decline_more_row_dims_than_the_grid_holds() {
    let hw = devices::m2_pro();
    let problem = problem(&[2, 2, 2, 2, 64], 4);
    let declines = |prepared: Result<(), cubecl::tile::launch::Refusal>| {
        matches!(
            prepared,
            Err(cubecl::tile::launch::Refusal::Declined { .. })
        )
    };
    let cube = CubeReduceRoutine.settings(&hw)[0];
    assert!(declines(
        CubeReduceRoutine.prepare(&hw, &problem, &cube).map(|_| ())
    ));
    let hypercube = HypercubeReduceRoutine.settings(&hw)[0];
    assert!(declines(
        HypercubeReduceRoutine
            .prepare(&hw, &problem, &hypercube)
            .map(|_| ())
    ));
    let unit = UnitReduceRoutine.settings(&hw)[0];
    assert!(UnitReduceRoutine.prepare(&hw, &problem, &unit).is_ok());
    let plane = PlaneReduceRoutine.settings(&hw)[0];
    assert!(PlaneReduceRoutine.prepare(&hw, &problem, &plane).is_ok());
}
