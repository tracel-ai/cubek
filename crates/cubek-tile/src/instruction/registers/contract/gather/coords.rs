//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::*;

use super::{GatherProblem, LhsRole};
use crate::instruction::registers::contract::resolve_nd_coords;

/// One operand read at the accumulator cell `(row, col)` of the batch matrix `batch` names.
/// `width` is the operand's own line width, since only its innermost axis is addressed in lines.
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

/// Assembles the accumulator cell coordinate [`resolve_nd_coords`] reads on its acc branch: one
/// entry per axis of the accumulator's space, in the space's own order, because that is what
/// `acc.position(axis)` indexes it by.
///
/// `batch`'s axes come already resolved; `row` unravels over the row group and `col` over the
/// column group, whose innermost entry counts the cells one block column holds rather than
/// scalars ([`ContractShape::cell_width`]). One axis per group gives `[batch…, row, col]`.
///
/// A column group spans several axes when `MatrixAxes::accumulator`, which stops it at the first
/// axis the lhs spans, finds none: a depthwise convolution's `[batch, out_h, out_w, channel]`
/// against a filter over channel and taps leaves them all in the column group.
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

/// What the separable schedule assumes on top of [`assert_operand_shapes`], where it steps one
/// resolved coordinate along the accumulator's columns by hand instead of resolving each cell.
///
/// [`Projection::validate`] already states this for an operand lined `contracted_per_step` wide,
/// but skips it at width `1`, where a scalar operand may gather on its innermost axis. The step
/// below needs it at every width, so it is asked here. Host-side, so a violation is comptime.
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

/// What [`resolve_nd_coords`] and the lane fold assume about how the operands are lined up: one
/// axis per operand is the vectorized one, addressed in lines; if it is not the axis the operand
/// really lines along, reads are silently off by the width. Host-side, so it fails at comptime.
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
    // `Space::contracted` merges lhs-first, so an axis only the rhs spans lands past every lhs
    // one and would take the `fastest` slot below. That reads as the lhs being lined wrong, which
    // it is not, so the real constraint is named here instead.
    for &axis in reduce {
        assert!(
            lhs.contains(axis),
            "contract gather: the lhs must span every contracted axis, but {axis:?} is contracted by \
             the rhs alone"
        );
    }
    let fastest = reduce[reduce.len() - 1];
    // A col-lined lhs is the exception: it lines along the accumulator's innermost axis, so the
    // fastest contracted axis is walked in elements like every other contracted one.
    assert!(
        lhs_role == LhsRole::LinedAlongColumn || lhs.axis_at(lhs.rank() - 1) == fastest,
        "contract gather: the lhs must line along the fastest contracted axis {fastest:?}"
    );
    // A vectorized rhs lines along the accumulator's innermost axis (its lanes are cells) or the
    // fastest contracted one (its lanes are partials of one cell); a scalar one is addressed in
    // elements and need not span either, so a weight shared by every column can omit that axis.
    let rhs_lined = rhs.axis_at(rhs.rank() - 1);
    assert!(
        rhs_vec_len == 1 || rhs_lined == acc.axis_at(acc.rank() - 1) || rhs_lined == fastest,
        "contract gather: a vectorized rhs must line along the accumulator's innermost axis or \
         the fastest contracted axis {fastest:?}"
    );
    // A [`LhsRole::PerCell`] lhs is read once per cell, and a cell is `rhs_vec_len` columns wide,
    // so one read covers them only when it lines along the column axis. Lined along a contracted
    // axis instead, the broadcast would silently serve the first column's value to every lane.
    assert!(
        lhs_role != LhsRole::PerCell || rhs_vec_len == 1,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis must be the one it lines along (the accumulator is {rhs_vec_len} \
         wide, the lhs lines {lhs_vec_len})"
    );
    // The col-lined line *is* the cell, so the two are the same width by construction.
    assert!(
        lhs_role != LhsRole::LinedAlongColumn || lhs_vec_len == rhs_vec_len,
        "contract gather: a col-lined lhs is read as the cell itself, so its line width \
         ({lhs_vec_len}) must be the accumulator's ({rhs_vec_len})"
    );
}
