//! What every reduce microkernel computes alone: each row's cells, one or `k` for a top-k, values
//! or indices. Along the reduced dimension, `input[BATCH, REDUCED]` and `out[BATCH, SLOTS]`; across
//! it, `input[REDUCED, BATCH]` and `out[SLOTS, BATCH]`. A hypercube's input splits the reduced
//! dimension into its parts, `input[BATCH, PARTS, REDUCED]` or `input[PARTS, REDUCED, BATCH]`, a
//! cube a part.

use crate::launch::reduce::{LineDirection, ReduceLauncher, TestedMicrokernel};
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Decline;
use cubecl::zspace::{Shape, shape};
use cubek_microkernel::reduce::{ReduceOperation, ReduceWorkItem, ReduceWrites};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// The reduce microkernel `hw` runs each case names.
pub(crate) struct ReduceMicrokernelScopeUnderTest {
    hw: Hardware,
}

/// One operation over rows `len` values long holding `rows`, read `width` values a line, laid
/// along or across the reduced dimension, reduced by `microkernel`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReduceCase {
    pub operation: ReduceOperation,
    pub microkernel: TestedMicrokernel,
    pub len: usize,
    pub width: usize,
    pub lines: LineDirection,
    pub rows: Rows,
}

/// What a case's rows hold, each kind a behavior of its own.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Rows {
    /// Small whole numbers, so every sum is exact, on a cycle of eleven that row `r` starts `r`
    /// values in and raises by `r`, so no two rows agree; for a product, `±1`, and `2` and `0.5`
    /// in turn on a cycle of nine, so no partial product of a long row overflows whatever order it
    /// multiplies in.
    Plain,
    /// Plain rows whose largest value sits at positions 1 and `len - 1` and smallest at 2 and
    /// `len - 2`: the lower index must win the tie.
    Ties,
    /// Plain rows with a NaN halfway and at the end.
    Nan,
    /// Every value `-inf`.
    NegativeInfinity,
    /// Every value zero.
    Zeros,
}

/// Rows a case lays along the reduced dimension.
const ROWS_ALONG: usize = 3;

/// The value a tie sits at, past every plain value.
const TIE: f32 = 1000.0;

impl Rows {
    /// Row `r` of `len` values, as `operation` reads it.
    fn row(self, operation: ReduceOperation, len: usize, r: usize) -> Vec<f32> {
        match self {
            Rows::Plain => plain_row(operation, len, r),
            Rows::Ties => with_values(plain_row(operation, len, r), &ties_of_len(len)),
            Rows::Nan => with_values(plain_row(operation, len, r), &nans_of_len(len)),
            Rows::NegativeInfinity => vec![f32::NEG_INFINITY; len],
            Rows::Zeros => vec![0.0; len],
        }
    }
}

/// Row `r` of [`Rows::Plain`], `len` values long, as `operation` reads it.
fn plain_row(operation: ReduceOperation, len: usize, r: usize) -> Vec<f32> {
    match operation {
        ReduceOperation::Prod => (r..r + len).map(product_value_of_index).collect(),
        _ => (r..r + len)
            .map(|i| ((i * 7) % 11) as f32 - 5.0 + r as f32)
            .collect(),
    }
}

/// Value `i` of the cycle a product's rows hold.
fn product_value_of_index(i: usize) -> f32 {
    match i % 9 {
        0 => 2.0,
        4 => 0.5,
        3 | 5 => -1.0,
        _ => 1.0,
    }
}

/// The positions and values of the ties of a row of `len` values.
fn ties_of_len(len: usize) -> [(usize, f32); 4] {
    [(1, TIE), (len - 1, TIE), (2, -TIE), (len - 2, -TIE)]
}

/// The positions and values of the NaN of a row of `len` values.
fn nans_of_len(len: usize) -> [(usize, f32); 2] {
    [len / 2, len - 1].map(|position| (position, f32::NAN))
}

/// `row` with each `(position, value)` of `at` written in.
fn with_values(mut row: Vec<f32>, at: &[(usize, f32)]) -> Vec<f32> {
    for &(position, value) in at {
        row[position] = value;
    }
    row
}

/// `case` launched once and checked against the reduce reference; a microkernel the device does
/// not run is a skip.
pub(crate) fn reduces_like_the_cpu(case: ReduceCase) {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    ReduceMicrokernelScopeUnderTest { hw }.check([case]);
}

impl ReduceCase {
    /// The output's element.
    fn out(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }
}

/// A reduction on the host: its rows and what each writes.
struct HostReduce {
    pub case: ReduceCase,
    /// How far apart the values its microkernel reads of a row lie.
    pub stride: usize,
    /// Each row's values, read in order and by row (Arrival and Position Usage).
    pub rows: Vec<Vec<f32>>,
}

impl HostReduce {
    /// The rows `case` names: [`ROWS_ALONG`] along the reduced dimension; across it, each row a
    /// column, as many as two whole lines.
    fn new(case: ReduceCase, hw: &Hardware) -> Self {
        let count = match case.lines {
            LineDirection::Along => ROWS_ALONG,
            LineDirection::Across => 2 * case.width,
        };
        let rows = (0..count)
            .map(|r| case.rows.row(case.operation, case.len, r))
            .collect();
        Self {
            case,
            stride: case.microkernel.stride_on(hw),
            rows,
        }
    }

    /// The input's shape and its values in that order.
    fn input(&self) -> (Shape, Vec<f32>) {
        match self.case.lines {
            LineDirection::Along => (self.input_along(), self.rows.concat()),
            LineDirection::Across => (self.input_across(), self.values_across()),
        }
    }

    /// The input's shape, its rows along the reduced dimension.
    fn input_along(&self) -> Shape {
        let (rows, len) = (self.rows.len(), self.case.len);
        match self.case.microkernel {
            TestedMicrokernel::HypercubeLastArrival { parts, .. } => {
                shape![rows, parts, len / parts]
            }
            _ => shape![rows, len],
        }
    }

    /// The input's shape, its rows across the reduced dimension.
    fn input_across(&self) -> Shape {
        let (rows, len) = (self.rows.len(), self.case.len);
        match self.case.microkernel {
            TestedMicrokernel::HypercubeLastArrival { parts, .. } => {
                shape![parts, len / parts, rows]
            }
            _ => shape![len, rows],
        }
    }

    /// The rows' values, a row a column.
    fn values_across(&self) -> Vec<f32> {
        (0..self.case.len)
            .flat_map(|r| self.rows.iter().map(move |row| row[r]))
            .collect()
    }

    /// The output's shape: a row's cells along its row, or down its column.
    fn out_shape(&self) -> Shape {
        let (rows, cells) = (self.rows.len(), self.case.operation.written_per_row());
        match self.case.lines {
            LineDirection::Along => shape![rows, cells],
            LineDirection::Across => shape![cells, rows],
        }
    }

    /// The values of `row` its microkernel reads.
    fn values_read(&self, row: &[f32]) -> Vec<f32> {
        row[..self.case.microkernel.values_of_len(row.len())]
            .iter()
            .step_by(self.stride)
            .copied()
            .collect()
    }

    /// What every row writes, laid as the output is.
    fn out(&self) -> HostData {
        let per_row: Vec<Vec<f32>> = self
            .rows
            .iter()
            .map(|row| self.case.operation.reference_cells(&self.values_read(row)))
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
        let host = HostReduce::new(*case, &self.hw);
        let item = ReduceWorkItem {
            operation: case.operation,
            input: F32,
            accumulator: F32,
            lines_along: case.lines.dimension(),
            with_indices: false,
        };
        let launch = ReduceLauncher::new(
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
        vec![HostReduce::new(*case, &self.hw).out()]
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-4
    }
}

/// A test for each `name: operation`, each one launch of `microkernel` on rows `len` values long
/// holding `rows`, read `width` values a line, laid `lines`.
macro_rules! reduce_tests {
    (
        $microkernel:expr,
        $lines:ident,
        $len:expr,
        $width:expr,
        $rows:ident,
        [$($name:ident: $operation:expr),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                $crate::reduce::reduces_like_the_cpu($crate::reduce::ReduceCase {
                    operation: $operation,
                    microkernel: $microkernel,
                    len: $len,
                    width: $width,
                    lines: $crate::launch::reduce::LineDirection::$lines,
                    rows: $crate::reduce::Rows::$rows,
                });
            }
        )*
    };
}

/// [`reduce_tests`] on plain rows, for every operation a reduce microkernel serves.
macro_rules! every_operation {
    ($microkernel:expr, $lines:ident, $len:expr, $width:expr) => {
        use cubek_microkernel::reduce::ReduceOperation;
        reduce_tests!($microkernel, $lines, $len, $width, Plain, [
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
            top_3: ReduceOperation::TopK { k: 3 },
            arg_top_3: ReduceOperation::ArgTopK { k: 3 },
        ]);
    };
}

/// [`reduce_tests`] on the rows whose values a reduction must rank or test with care, each kind
/// with the operations it matters to.
macro_rules! edge_rows {
    ($microkernel:expr, $lines:ident, $len:expr, $width:expr) => {
        /// The largest and the smallest value twice: the lower index wins.
        mod ties {
            use super::*;
            use cubek_microkernel::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, Ties, [
                arg_max: ReduceOperation::ArgMax,
                arg_min: ReduceOperation::ArgMin,
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        }

        /// A NaN twice: it ranks ahead of every number, so an extremum keeps the first.
        mod nan {
            use super::*;
            use cubek_microkernel::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, Nan, [
                max: ReduceOperation::Max,
                min: ReduceOperation::Min,
                max_abs: ReduceOperation::MaxAbs,
                arg_max: ReduceOperation::ArgMax,
                arg_min: ReduceOperation::ArgMin,
                top_3: ReduceOperation::TopK { k: 3 },
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        }

        /// Every value `-inf`: a real value, which outranks every empty slot.
        mod negative_infinity {
            use super::*;
            use cubek_microkernel::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, NegativeInfinity, [
                max: ReduceOperation::Max,
                arg_max: ReduceOperation::ArgMax,
                top_3: ReduceOperation::TopK { k: 3 },
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        }

        /// Every value zero: neither any nor all of them is set.
        mod zeros {
            use super::*;
            use cubek_microkernel::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, Zeros, [
                any: ReduceOperation::Any,
                all: ReduceOperation::All,
            ]);
        }
    };
}
