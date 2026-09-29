//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::*;

use super::base::{GatherProblem, LhsRole};
use crate::ops::matmul::leaf::memory::resolve_nd_coords;

/// One operand read at accumulator cell `(row, col)` of batch matrix `batch`.
#[cube]
pub(super) fn cell_read<T: Numeric, W: Size>(
    view: &Masked<'_, Vector<T, W>, CoordsDyn>,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] operand: Space,
    #[comptime] problem: GatherProblem,
    #[comptime] width: usize,
) -> Vector<T, W> {
    view.read(cell_position(
        batch,
        row,
        col,
        reduce_coords,
        operand,
        problem,
        width,
    ))
}

/// Resolve one operand's N-D coordinate from an accumulator cell and contracted coordinates.
#[cube]
pub(super) fn cell_position(
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] operand: Space,
    #[comptime] problem: GatherProblem,
    #[comptime] width: usize,
) -> CoordsDyn {
    let acc_coords = acc_cell_coords(
        batch,
        row,
        col,
        comptime!(problem.block.row_extents()),
        comptime!(problem.block.column_line_extents()),
    );
    resolve_nd_coords(
        operand,
        comptime!(problem.block.space.clone()),
        comptime!(problem.block.reduce.clone()),
        &acc_coords,
        reduce_coords,
        width,
        false,
    )
}

/// `coords` with its last entry offset by `delta`.
#[cube]
pub(super) fn offset_last(coords: &CoordsDyn, #[comptime] rank: usize, delta: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        out.push(if comptime!(p == rank - 1) {
            coords[p].plus(delta)
        } else {
            coords[p]
        });
    }
    out
}

/// The accumulator cell coordinate [`resolve_nd_coords`] reads, one entry per axis of the
/// accumulator's space in its own order.
#[cube]
fn acc_cell_coords(
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] row_extents: Vec<usize>,
    #[comptime] col_extents: Vec<usize>,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..batch.len() {
        out.push(batch.at(p));
    }
    let rows = Coords::constant(comptime!(row_extents.clone())).unravel(row);
    #[unroll]
    for p in 0..rows.len() {
        out.push(rows.at(p));
    }
    let cols = Coords::constant(comptime!(col_extents.clone())).unravel(col);
    #[unroll]
    for p in 0..cols.len() {
        out.push(cols.at(p));
    }

    out
}

/// Assert the extra operand shapes the separable schedule assumes; fails at comptime.
pub(super) fn assert_separable_shapes(rhs: &Projection, acc: &Space, rhs_spans_col: bool) {
    let col = acc.axis_at(acc.rank() - 1);
    assert!(
        rhs_spans_col,
        "contract gather: the separable schedule walks the accumulator's columns by stepping the \
         rhs's innermost physical axis, so the rhs must span {col:?}"
    );
    let innermost = rhs.physical_axis(rhs.physical_rank() - 1);
    assert!(
        innermost.is_identity(col),
        "contract gather: the separable schedule steps the rhs's innermost physical axis once per \
         accumulator column, so that axis must be {col:?} at coefficient 1"
    );
}

/// Assert each operand's vectorized axis is the one it lines along; fails at comptime.
#[allow(clippy::too_many_arguments)]
pub(super) fn assert_operand_shapes(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    reduce: &[Axis],
    lhs_vec_len: usize,
    rhs_vec_len: usize,
    lhs_role: LhsRole,
) {
    assert!(
        !reduce.is_empty(),
        "contract gather: the operands contract no axis against the accumulator"
    );
    // An rhs-only contracted axis would take the `fastest` slot; name that constraint instead.
    for &axis in reduce {
        assert!(
            lhs.contains(axis),
            "contract gather: the lhs must span every contracted axis, but {axis:?} is contracted by \
             the rhs alone"
        );
    }
    let fastest = reduce[reduce.len() - 1];
    // A col-lined lhs lines along the accumulator, so every contracted axis is walked in elements.
    assert!(
        lhs_role == LhsRole::LinedAlongColumn || lhs.axis_at(lhs.rank() - 1) == fastest,
        "contract gather: the lhs must line along the fastest contracted axis {fastest:?}"
    );
    // A scalar rhs need not span either axis, so a weight shared by every column can omit it.
    let rhs_lined = rhs.axis_at(rhs.rank() - 1);
    assert!(
        rhs_vec_len == 1 || rhs_lined == acc.axis_at(acc.rank() - 1) || rhs_lined == fastest,
        "contract gather: a vectorized rhs must line along the accumulator's innermost axis or \
         the fastest contracted axis {fastest:?}"
    );
    // A per-cell lhs read covers `rhs_vec_len` columns only when lined along the column axis.
    assert!(
        lhs_role != LhsRole::PerCell || rhs_vec_len == 1,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis must be the one it lines along (the accumulator is {rhs_vec_len} \
         wide, the lhs lines {lhs_vec_len})"
    );
    assert!(
        lhs_role != LhsRole::LinedAlongColumn || lhs_vec_len == rhs_vec_len,
        "contract gather: a col-lined lhs is read as the cell itself, so its line width \
         ({lhs_vec_len}) must be the accumulator's ({rhs_vec_len})"
    );
}
