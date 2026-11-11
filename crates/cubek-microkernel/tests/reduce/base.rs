//! What every reduce microkernel computes alone: each row's cells, one or `k` for a top-k, values
//! or indices. Along the reduced axis, `input[o, r]` and `out[o, s]`; across it, `input[r, o]` and
//! `out[s, o]`. A hypercube's input splits the reduced axis into its parts, `input[o, p, r]` or
//! `input[p, r, o]`, a cube a part.

use crate::launch::reduce::{ReduceLaunch, ReduceScope};
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation, ReduceWorkItem, ReduceWrites};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// Every operation a reduce microkernel serves.
pub(crate) const OPERATIONS: [ReduceOperation; 12] = [
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

/// The reduce microkernel `hw` runs at each case's scope.
pub(crate) struct Reduce {
    pub hw: Hardware,
}

/// One operation over rows `len` values long, read `width` values a line, laid along or across
/// the reduced axis, reduced by `scope`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReduceCase {
    pub operation: ReduceOperation,
    pub scope: ReduceScope,
    pub len: usize,
    pub width: usize,
    pub layout: ReduceLayout,
}

impl ReduceCase {
    /// Every operation over `(len, width)` rows reduced by `scope` along the reduced axis, then
    /// across it at `across`.
    pub(crate) fn every_operation(
        scope: ReduceScope,
        along: &[(usize, usize)],
        across: &[(usize, usize)],
    ) -> Vec<Self> {
        let laid = |layout, sizes: &[(usize, usize)]| {
            sizes
                .iter()
                .flat_map(move |&(len, width)| {
                    OPERATIONS.map(|operation| Self {
                        operation,
                        scope,
                        len,
                        width,
                        layout,
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut cases = laid(ReduceLayout::Along, along);
        cases.extend(laid(ReduceLayout::Across, across));
        cases
    }

    /// The element the output is written in: the values', or an index's.
    fn out(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }
}

/// A reduction on the host: its rows and what each writes.
struct HostReduce {
    case: ReduceCase,
    rows: Vec<Vec<f32>>,
}

impl HostReduce {
    /// Rows of `len` values: small whole numbers, so every sum is exact; one row with ties at
    /// both ends, one with a NaN, one all `-inf`, one with zeros; for a product, values `±1`, and
    /// `2` and `0.5` in turn, so no partial product of a long row overflows, whatever order it
    /// multiplies in. Across the reduced axis, every row a column: as many rows as a whole number
    /// of lines.
    fn new(case: ReduceCase) -> Self {
        let len = case.len;
        let base: Vec<f32> = match case.operation {
            ReduceOperation::Prod => (0..len)
                .map(|i| match i % 9 {
                    0 => 2.0,
                    4 => 0.5,
                    3 | 5 => -1.0,
                    _ => 1.0,
                })
                .collect(),
            _ => (0..len).map(|i| ((i * 7) % 11) as f32 - 5.0).collect(),
        };
        let with = |at: &[(usize, f32)]| {
            let mut row = base.clone();
            for &(i, v) in at {
                row[i % len] = v;
            }
            row
        };
        let rows = vec![
            base.clone(),
            with(&[(1, 9.0), (len - 1, 9.0), (2, -9.0), (len - 2, -9.0)]),
            with(&[(len / 2, f32::NAN), (len - 1, f32::NAN)]),
            vec![f32::NEG_INFINITY; len],
            with(&[(0, 0.0), (len / 3, 0.0)]),
        ];
        let rows = match case.layout {
            ReduceLayout::Along => rows,
            ReduceLayout::Across => rows.iter().cycle().take(2 * case.width).cloned().collect(),
        };
        Self { case, rows }
    }

    /// The input's shape and its values in that order.
    fn input(&self) -> (Shape, Vec<f32>) {
        let (rows, len) = (self.rows.len(), self.case.len);
        match (self.case.layout, self.case.scope) {
            (ReduceLayout::Along, ReduceScope::Hypercube { parts, .. }) => {
                (shape![rows, parts, len / parts], self.rows.concat())
            }
            (ReduceLayout::Along, _) => (shape![rows, len], self.rows.concat()),
            (ReduceLayout::Across, scope) => {
                let shape = match scope {
                    ReduceScope::Hypercube { parts, .. } => shape![parts, len / parts, rows],
                    _ => shape![len, rows],
                };
                let values = (0..len)
                    .flat_map(|r| self.rows.iter().map(move |row| row[r]))
                    .collect();
                (shape, values)
            }
        }
    }

    /// The output's shape: a row's cells along its row, or down its column.
    fn out_shape(&self) -> Shape {
        let (rows, cells) = (self.rows.len(), self.case.operation.written_per_row());
        match self.case.layout {
            ReduceLayout::Along => shape![rows, cells],
            ReduceLayout::Across => shape![cells, rows],
        }
    }

    /// What every row writes, laid as the output is.
    fn out(&self) -> HostData {
        let per_row: Vec<Vec<f32>> = self
            .rows
            .iter()
            .map(|row| Self::cells(row, self.case.operation))
            .collect();
        let values = match self.case.layout {
            ReduceLayout::Along => per_row.concat(),
            ReduceLayout::Across => (0..self.case.operation.written_per_row())
                .flat_map(|s| per_row.iter().map(move |cells| cells[s]))
                .collect(),
        };
        HostData::row_major(HostDataVec::F32(values), self.out_shape())
    }

    /// What `operation` writes for `row`: its cells, values or indices as numbers.
    fn cells(row: &[f32], operation: ReduceOperation) -> Vec<f32> {
        let nan = row.iter().any(|v| v.is_nan());
        let reduce_row = |init: f32, f: fn(f32, f32) -> f32| row.iter().copied().fold(init, f);
        let keeping_nan = |v: f32| if nan { f32::NAN } else { v };
        let flag = |holds: bool| if holds { 1.0 } else { 0.0 };
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
            ReduceOperation::Any => vec![flag(row.iter().any(|v| *v != 0.0))],
            ReduceOperation::All => vec![flag(row.iter().all(|v| *v != 0.0))],
            ReduceOperation::ArgMax => vec![Self::ranked(row, true, 1)[0].1 as f32],
            ReduceOperation::ArgMin => vec![Self::ranked(row, false, 1)[0].1 as f32],
            ReduceOperation::TopK { k } => Self::ranked(row, true, k).iter().map(|e| e.0).collect(),
            ReduceOperation::ArgTopK { k } => Self::ranked(row, true, k)
                .iter()
                .map(|e| e.1 as f32)
                .collect(),
        }
    }

    /// The `k` best of `row` under `largest`, with their indices: the larger (smaller) value
    /// first, the lower index between equal values, a NaN ahead of every number.
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
}

impl ScopeUnderTest for Reduce {
    type Case = ReduceCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let host = HostReduce::new(*case);
        let item = ReduceWorkItem {
            operation: case.operation,
            input: F32,
            accumulator: F32,
            layout: case.layout,
            with_indices: false,
        };
        let launch = ReduceLaunch::new(
            &self.hw,
            item,
            case.scope,
            host.rows.len(),
            case.len,
            case.width,
            case.out(),
        )?;
        let (input_shape, input_values) = host.input();
        let out_shape = host.out_shape();
        let cells = out_shape.num_elements();
        let input = TestInput::builder(client.clone(), input_shape)
            .dtype(F32)
            .custom(input_values)
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), out_shape)
            .dtype(case.out())
            .custom(vec![-7.0; cells])
            .generate_without_host_data();
        launch.launch(
            client,
            input.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![HostReduce::new(*case).out()]
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-4
    }
}
