use crate::reduce::testing::{LineDirection, MicrokernelLauncher, Rows};
use crate::reduce::{ReduceOperation, ReduceWrites};
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::zspace::{Shape, shape};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataVec};

pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// Rows a case lays along the reduced dimension.
const ROWS_ALONG: usize = 3;

/// One operation over rows `len` values long holding `rows`, read `width` values a line, laid
/// along or across the reduced dimension, reduced by `microkernel`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReduceCase<M> {
    pub operation: ReduceOperation,
    pub microkernel: M,
    /// Values a row holds.
    pub len: usize,
    pub width: usize,
    pub lines: LineDirection,
    pub rows: Rows,
}

impl<M> ReduceCase<M> {
    /// The output's element.
    pub(crate) fn out(&self) -> ElemType {
        match self.operation.writes() {
            ReduceWrites::Values | ReduceWrites::Both => F32,
            ReduceWrites::Indices => I32,
        }
    }
}

/// A reduction on the host: its rows and what each writes.
pub(crate) struct HostReduce<M> {
    pub case: ReduceCase<M>,
    /// How far apart the values its microkernel reads of a row lie.
    pub stride: usize,
    /// Each row's values, read in order and by row (Arrival and Position Usage).
    pub rows: Vec<Vec<f32>>,
}

impl<M: MicrokernelLauncher> HostReduce<M> {
    /// The rows `case` names: [`ROWS_ALONG`] along the reduced dimension; across it, each row a
    /// column, as many as two whole lines.
    pub(crate) fn new(case: ReduceCase<M>, hw: &Hardware) -> Self {
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
    pub(crate) fn input(&self) -> (Shape, Vec<f32>) {
        match self.case.lines {
            LineDirection::Along => (self.input_along(), self.rows.concat()),
            LineDirection::Across => (self.input_across(), self.values_across()),
        }
    }

    /// The input's shape, its rows along the reduced dimension, split into their parts where there are
    /// several.
    fn input_along(&self) -> Shape {
        let (rows, len) = (self.rows.len(), self.case.len);
        match self.case.microkernel.parts() {
            1 => shape![rows, len],
            parts => shape![rows, parts, len / parts],
        }
    }

    /// The input's shape, its rows across the reduced dimension.
    fn input_across(&self) -> Shape {
        let (rows, len) = (self.rows.len(), self.case.len);
        match self.case.microkernel.parts() {
            1 => shape![len, rows],
            parts => shape![parts, len / parts, rows],
        }
    }

    /// The rows' values, a row a column.
    fn values_across(&self) -> Vec<f32> {
        (0..self.case.len)
            .flat_map(|r| self.rows.iter().map(move |row| row[r]))
            .collect()
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
        row[..self.case.microkernel.values_of_len(row.len())]
            .iter()
            .step_by(self.stride)
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
            LineDirection::Across => (0..self.case.operation.written_per_row())
                .flat_map(|s| per_row.iter().map(move |cells| cells[s]))
                .collect(),
        };
        HostData::row_major(HostDataVec::F32(values), self.out_shape())
    }
}
