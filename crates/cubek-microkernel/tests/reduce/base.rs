//! What every reduce microkernel computes alone: each row's cells, one or `k` for a top-k, values
//! or indices. Along the reduced dimension, `input[BATCH, REDUCED]` and `out[BATCH, SLOTS]`; across
//! it, `input[REDUCED, BATCH]` and `out[SLOTS, BATCH]`. A hypercube's input splits the reduced
//! dimension into its parts, `input[BATCH, PARTS, REDUCED]` or `input[PARTS, REDUCED, BATCH]`, a
//! cube a part.

use crate::launch::reduce::{LineDirection, ReduceLaunch, TestedMicrokernel};
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Decline;
use cubecl::zspace::{Shape, shape};
use cubek_microkernel::reduce::{ReduceOperation, ReduceWorkItem, ReduceWrites};
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

/// The reduce microkernel `hw` runs each case names.
pub(crate) struct ReduceMicrokernelScopeUnderTest {
    pub hw: Hardware,
}

/// One operation over rows `len` values long, read `width` values a line, laid along or across
/// the reduced dimension, reduced by `microkernel`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReduceCase {
    pub operation: ReduceOperation,
    pub microkernel: TestedMicrokernel,
    pub len: usize,
    pub width: usize,
    pub lines: LineDirection,
}

impl ReduceCase {
    /// Every operation over `(len, width)` rows reduced by `microkernel` along the reduced dimension,
    /// then across it at `across`.
    pub(crate) fn every_operation(
        microkernel: TestedMicrokernel,
        along: &[(usize, usize)],
        across: &[(usize, usize)],
    ) -> Vec<Self> {
        let laid = |lines, sizes: &[(usize, usize)]| {
            sizes
                .iter()
                .flat_map(move |&(len, width)| {
                    OPERATIONS.map(|operation| Self {
                        operation,
                        microkernel,
                        len,
                        width,
                        lines,
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut cases = laid(LineDirection::Along, along);
        cases.extend(laid(LineDirection::Across, across));
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
    /// Each row's values, read in order and by row (Arrival and Position Usage).
    rows: Vec<Vec<f32>>,
}

impl HostReduce {
    /// Rows of `len` values: small whole numbers, so every sum is exact; one row with ties at both
    /// ends, one with a NaN, one all `-inf`, one with zeros; for a product, values `±1`, and `2`
    /// and `0.5` in turn, so no partial product of a long row overflows, whatever order it
    /// multiplies in. Across the reduced dimension, every row a column: as many rows as a whole
    /// number of lines.
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
        let rows = match case.lines {
            LineDirection::Along => rows,
            LineDirection::Across => rows.iter().cycle().take(2 * case.width).cloned().collect(),
        };
        Self { case, rows }
    }

    /// The input's shape and its values in that order.
    fn input(&self) -> (Shape, Vec<f32>) {
        let (rows, len) = (self.rows.len(), self.case.len);
        match (self.case.lines, self.case.microkernel) {
            (LineDirection::Along, TestedMicrokernel::HypercubeLastArrival { parts, .. }) => {
                (shape![rows, parts, len / parts], self.rows.concat())
            }
            (LineDirection::Along, _) => (shape![rows, len], self.rows.concat()),
            (LineDirection::Across, microkernel) => {
                let shape = match microkernel {
                    TestedMicrokernel::HypercubeLastArrival { parts, .. } => {
                        shape![parts, len / parts, rows]
                    }
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
        match self.case.lines {
            LineDirection::Along => shape![rows, cells],
            LineDirection::Across => shape![cells, rows],
        }
    }

    /// What every row writes, laid as the output is.
    fn out(&self) -> HostData {
        let per_row: Vec<Vec<f32>> = self
            .rows
            .iter()
            .map(|row| self.case.operation.reference_cells(row))
            .collect();
        let values = match self.case.lines {
            LineDirection::Along => per_row.concat(),
            LineDirection::Across => (0..self.case.operation.written_per_row())
                .flat_map(|s| per_row.iter().map(move |cells| cells[s]))
                .collect(),
        };
        HostData::row_major(HostDataVec::F32(values), self.out_shape())
    }
}

impl ScopeUnderTest for ReduceMicrokernelScopeUnderTest {
    type Case = ReduceCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Decline> {
        let host = HostReduce::new(*case);
        let item = ReduceWorkItem {
            operation: case.operation,
            input: F32,
            accumulator: F32,
            lines_along: case.lines.dimension(),
            with_indices: false,
        };
        let launch = ReduceLaunch::new(
            &self.hw,
            item,
            case.microkernel,
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
