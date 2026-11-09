//! The reduce microkernels launched on their own on the test device, each operation against a CPU
//! reference: rows along the reduced axis and across it, at one and four values a line, in every
//! scope.

use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::zspace::shape;
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation, ReduceWorkItem, ReduceWrites};
use cubek_microkernel::testing::reduce::{ReduceLaunch, ReduceScope};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, TestInput};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// The `k` best of `row` under `largest`, with their indices: the larger (smaller) value first,
/// the lower index between equal values, a NaN ahead of every number.
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

/// What `operation` writes for `row`: its cells, values or indices as numbers.
fn reference(row: &[f32], operation: ReduceOperation) -> Vec<f32> {
    let nan = row.iter().any(|v| v.is_nan());
    let reduce_row = |init: f32, f: fn(f32, f32) -> f32| row.iter().copied().fold(init, f);
    let keeping_nan = |v: f32| if nan { f32::NAN } else { v };
    match operation {
        ReduceOperation::Sum => vec![row.iter().sum()],
        ReduceOperation::Prod => vec![row.iter().product()],
        ReduceOperation::Mean => vec![row.iter().sum::<f32>() / row.len() as f32],
        ReduceOperation::Max => vec![keeping_nan(reduce_row(f32::NEG_INFINITY, f32::max))],
        ReduceOperation::Min => vec![keeping_nan(reduce_row(f32::INFINITY, f32::min))],
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

/// Rows of `len` values: small whole numbers, so every sum is exact; one row with ties at both
/// ends, one with a NaN, one all `-inf`, one with zeros; for a product, values `±1`, and `2` and
/// `0.5` in turn, so no partial product of a long row overflows, whatever order it multiplies in.
fn rows(len: usize, product: bool) -> Vec<Vec<f32>> {
    let base: Vec<f32> = match product {
        true => (0..len)
            .map(|i| match i % 9 {
                0 => 2.0,
                4 => 0.5,
                3 | 5 => -1.0,
                _ => 1.0,
            })
            .collect(),
        false => (0..len).map(|i| ((i * 7) % 11) as f32 - 5.0).collect(),
    };
    let with = |at: &[(usize, f32)]| {
        let mut row = base.clone();
        for &(i, v) in at {
            row[i % len] = v;
        }
        row
    };
    vec![
        base.clone(),
        with(&[(1, 9.0), (len - 1, 9.0), (2, -9.0), (len - 2, -9.0)]),
        with(&[(len / 2, f32::NAN), (len - 1, f32::NAN)]),
        vec![f32::NEG_INFINITY; len],
        with(&[(0, 0.0), (len / 3, 0.0)]),
    ]
}

fn same(got: f32, want: f32) -> bool {
    (got.is_nan() && want.is_nan()) || got == want || (got - want).abs() <= 1e-4 * want.abs()
}

fn check(hw: &Hardware, operation: ReduceOperation, scope: ReduceScope, len: usize, width: usize) {
    check_laid(hw, operation, scope, len, width, ReduceLayout::Along);
}

/// [`check`] with the rows laid as `layout` says: along, `input[o, r]`; across, `input[r, o]`.
fn check_laid(
    hw: &Hardware,
    operation: ReduceOperation,
    scope: ReduceScope,
    len: usize,
    width: usize,
    layout: ReduceLayout,
) {
    let client = cubecl::test_device().client();
    let item = ReduceWorkItem {
        operation,
        input: F32,
        accumulator: F32,
        layout,
        with_indices: false,
    };
    let out_elem = match operation.writes() {
        ReduceWrites::Values | ReduceWrites::Both => F32,
        ReduceWrites::Indices => I32,
    };
    let rows = rows(len, operation == ReduceOperation::Prod);
    let cells = operation.written_per_row();
    // Across, every row a column: as many rows as a whole number of lines.
    let rows: Vec<Vec<f32>> = match layout {
        ReduceLayout::Along => rows,
        ReduceLayout::Across => rows.iter().cycle().take(2 * width).cloned().collect(),
    };
    let launch = ReduceLaunch::new(hw, item, scope, rows.len(), len, width, out_elem)
        .unwrap_or_else(|why| panic!("{operation:?} {scope:?}: {why:?}"));
    // A hypercube's input is the same values with the reduced axis split into its parts.
    let (input_shape, input_values, out_shape) = match layout {
        ReduceLayout::Along => (
            match scope {
                ReduceScope::Hypercube { parts, .. } => shape![rows.len(), parts, len / parts],
                _ => shape![rows.len(), len],
            },
            rows.concat(),
            shape![rows.len(), cells],
        ),
        ReduceLayout::Across => (
            match scope {
                ReduceScope::Hypercube { parts, .. } => shape![parts, len / parts, rows.len()],
                _ => shape![len, rows.len()],
            },
            (0..len)
                .flat_map(|r| rows.iter().map(move |row| row[r]))
                .collect(),
            shape![cells, rows.len()],
        ),
    };
    let input = TestInput::builder(client.clone(), input_shape)
        .dtype(F32)
        .custom(input_values)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), out_shape)
        .dtype(out_elem)
        .custom(vec![-7.0; rows.len() * cells])
        .generate_without_host_data();
    launch
        .launch(
            &client,
            input.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        )
        .unwrap_or_else(|why| panic!("{operation:?} {scope:?}: {why:?}"));
    let host = match out_elem {
        I32 => HostDataType::I32,
        _ => HostDataType::F32,
    };
    let got = HostData::from_tensor_handle(&client, out, host);
    for (o, row) in rows.iter().enumerate() {
        for (s, want) in reference(row, operation).into_iter().enumerate() {
            let at = match layout {
                ReduceLayout::Along => [o, s],
                ReduceLayout::Across => [s, o],
            };
            let have = match out_elem {
                I32 => got.get_i32(&at) as f32,
                _ => got.get_f32(&at),
            };
            assert!(
                same(have, want),
                "{operation:?} {scope:?} {layout:?} len {len} width {width} row {o} cell {s}: got \
                 {have}, want {want}"
            );
        }
    }
}

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

/// A unit alone reduces every operation to the CPU's answer.
#[test]
fn a_unit_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    for operation in OPERATIONS {
        for (len, width) in [(37, 1), (64, 4)] {
            check(&hw, operation, ReduceScope::Unit, len, width);
        }
    }
}

/// A plane reduces every operation to the CPU's answer, rows shorter and longer than its lanes.
#[test]
fn a_plane_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    if hw.plane_dim().is_err() {
        return;
    }
    for operation in OPERATIONS {
        for (len, width) in [(37, 1), (20, 4), (1024, 4)] {
            check(&hw, operation, ReduceScope::Plane, len, width);
        }
    }
}

/// A cube of four planes reduces every operation to the CPU's answer, rows shorter than its units
/// and several times longer.
#[test]
fn a_cube_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    if hw.plane_dim().is_err() {
        return;
    }
    for operation in OPERATIONS {
        for (len, width) in [(37, 1), (4096, 4)] {
            check(&hw, operation, ReduceScope::Cube { planes: 4 }, len, width);
        }
    }
}

/// Across the reduced axis, each lane its own row, the unit, the plane and the cube reduce every
/// operation to the CPU's answer.
#[test]
fn every_scope_reduces_rows_side_by_side_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let mut scopes = vec![ReduceScope::Unit];
    if hw.plane_dim().is_ok() {
        scopes.extend([ReduceScope::Plane, ReduceScope::Cube { planes: 4 }]);
    }
    for scope in scopes {
        for operation in OPERATIONS {
            for (len, width) in [(37, 1), (300, 4)] {
                check_laid(&hw, operation, scope, len, width, ReduceLayout::Across);
            }
        }
    }
}

/// A row split across cubes, four planes each, reduces every operation to the CPU's answer: along
/// the reduced axis, over two parts and over eight, a part shorter than the cube's units and one
/// several times longer; and across it, each lane its own row.
#[test]
fn a_hypercube_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    // The cubes meet through device memory, which the device must keep coherent within a launch.
    if hw.plane_dim().is_err() || !hw.properties.features.device_memory_scope {
        return;
    }
    for operation in OPERATIONS {
        for (len, width, parts) in [(64, 1, 2), (8192, 4, 8)] {
            let scope = ReduceScope::Hypercube { planes: 4, parts };
            check(&hw, operation, scope, len, width);
        }
        let scope = ReduceScope::Hypercube {
            planes: 4,
            parts: 3,
        };
        check_laid(&hw, operation, scope, 300, 4, ReduceLayout::Across);
    }
}
