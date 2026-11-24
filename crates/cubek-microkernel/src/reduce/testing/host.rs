use crate::reduce::testing::{LineDirection, Rows};
use crate::reduce::{ReduceOperation, ReduceWrites};
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::zspace::{Shape, shape};
use cubek_test_utils::{HostData, HostDataVec};

pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// Rows a case lays along the reduced dimension.
const ROW_ALONG_COUNT: usize = 3;

/// One operation over rows `len` values long holding `rows`, read `width` values a line, laid
/// along or across the reduced dimension, reduced by `microkernel`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReduceCase {
    pub operation: ReduceOperation,
    /// Values a row holds.
    pub value_count: usize,
    /// Values a line holds.
    pub lane_count: usize,
    pub lines: LineDirection,
    pub rows: Rows,
}

impl ReduceCase {
    /// The output's element.
    pub(crate) fn out(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }
}

/// What a microkernel reads of a row, as its launch lays it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowShare {
    /// The parts a row is laid in.
    pub part_count: usize,
    /// How many values apart those it reads lie.
    pub stride_count: usize,
    /// The values it reads among.
    pub value_count: usize,
}

/// A reduction on the host: its rows and what each writes.
pub(crate) struct HostReduce {
    pub case: ReduceCase,
    pub share: RowShare,
    /// Each row's values, read in order and by row (Arrival and Position Usage).
    pub rows: Vec<Vec<f32>>,
}

impl HostReduce {
    /// The rows `case` names: [`ROW_ALONG_COUNT`] along the reduced dimension; across it, each row a
    /// column, as many as two whole lines.
    pub(crate) fn new(case: ReduceCase, share: RowShare) -> Self {
        let count = match case.lines {
            LineDirection::Along => ROW_ALONG_COUNT,
            LineDirection::Across => 2 * case.lane_count,
        };
        let rows = (0..count)
            .map(|r| case.rows.row(case.operation, case.value_count, r))
            .collect();
        Self { case, share, rows }
    }

    /// The input's shape and its values in that order.
    pub(crate) fn input(&self) -> (Shape, Vec<f32>) {
        match self.case.lines {
            LineDirection::Along => (self.input_along(), self.rows.concat()),
            LineDirection::Across => (self.input_across(), self.values_across()),
        }
    }

    /// The input's shape, its rows along the reduced dimension, split into their parts where there are
    /// several.
    fn input_along(&self) -> Shape {
        let (rows, len) = (self.rows.len(), self.case.value_count);
        match self.share.part_count {
            1 => shape![rows, len],
            parts => shape![rows, parts, len / parts],
        }
    }

    /// The input's shape, its rows across the reduced dimension.
    fn input_across(&self) -> Shape {
        let (rows, len) = (self.rows.len(), self.case.value_count);
        match self.share.part_count {
            1 => shape![len, rows],
            parts => shape![parts, len / parts, rows],
        }
    }

    /// The rows' values, a row a column.
    fn values_across(&self) -> Vec<f32> {
        side_by_side(&self.rows, self.case.value_count)
    }

    /// The output's shape: a row's cells along its row, or down its column.
    pub(crate) fn out_shape(&self) -> Shape {
        let (rows, cells) = (self.rows.len(), self.case.operation.written_per_row());
        match self.case.lines {
            LineDirection::Along => shape![rows, cells],
            LineDirection::Across => shape![cells, rows],
        }
    }

    /// The values of `row` its microkernel reads.
    fn values_read(&self, row: &[f32]) -> Vec<f32> {
        row[..self.share.value_count]
            .iter()
            .step_by(self.share.stride_count)
            .copied()
            .collect()
    }

    /// What every row writes, laid as the output is.
    pub(crate) fn out(&self) -> HostData {
        let per_row: Vec<Vec<f32>> = self
            .rows
            .iter()
            .map(|row| self.case.operation.reference_cells(&self.values_read(row)))
            .collect();
        let values = match self.case.lines {
            LineDirection::Along => per_row.concat(),
            LineDirection::Across => side_by_side(&per_row, self.case.operation.written_per_row()),
        };
        HostData::row_major(HostDataVec::F32(values), self.out_shape())
    }
}

/// `rows`, `value_count` values each, a row a column.
fn side_by_side(rows: &[Vec<f32>], value_count: usize) -> Vec<f32> {
    (0..value_count)
        .flat_map(|at| rows.iter().map(move |row| row[at]))
        .collect()
}
