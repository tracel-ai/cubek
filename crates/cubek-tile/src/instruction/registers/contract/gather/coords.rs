//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::*;

use super::{GatherProblem, LhsRole};

/// One operand read at the accumulator cell `(row, col)` of the batch matrix `batch` names.
/// `width` is the operand's own line width, since only its innermost axis is addressed in lines.
#[cube]
pub(super) fn cell_read<T: Numeric, W: Size>(
    view: &MaskedView<'_, Vector<T, W>, CoordsDyn>,
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
    let acc_coords = acc_cell_coords(batch, row, col);
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
            coords[p].fadd(delta)
        } else {
            coords[p]
        });
    }
    out
}

/// Assembles the accumulator cell coordinate [`resolve_nd_coords`] reads on its acc branch:
/// `batch`'s own axes in order, then `row` (the accumulator's second to last axis), then `col`
/// (its last). This is the axis order [`resolve_nd_coords`] assumes when it looks up
/// `acc.position(axis)`.
#[cube]
fn acc_cell_coords(batch: &Coords<u32>, row: u32, col: u32) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..batch.len() {
        out.push(batch.at(p));
    }
    out.push(row);
    out.push(col);

    out
}

/// What the separable schedule assumes on top of [`assert_operand_shapes`], where it steps one
/// resolved coordinate along the accumulator's columns by hand instead of resolving each cell.
///
/// [`Projection::validate`] already states this for an operand served in lines, but it skips the
/// rule at width `1`, where there are no lines to address and a scalar operand is free to gather
/// on its innermost axis. The step below is not free of it either way, so it is asked for here at
/// every width. Host-side, so a violation is a comptime message.
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

/// What [`resolve_nd_coords`] and the lane fold assume about how the operands are lined up. Both
/// treat one axis per operand as the vectorized one and address it in lines; if that is not the
/// axis the operand actually lines along, the reads are silently off by the width rather than
/// wrong in a way a test would localize. Host-side, so a violation is a comptime message.
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
    // elements and need not span either, which is what lets a weight shared by every column live
    // in a space that simply omits it.
    let rhs_lined = rhs.axis_at(rhs.rank() - 1);
    assert!(
        rhs_vec_len == 1 || rhs_lined == acc.axis_at(acc.rank() - 1) || rhs_lined == fastest,
        "contract gather: a vectorized rhs must line along the accumulator's innermost axis or \
         the fastest contracted axis {fastest:?}"
    );
    // A [`LhsRole::PerCell`] lhs is read once per cell, and a cell is `rhs_vec_len` columns wide.
    // One read covers them only when the column is the axis it lines along, which is the case the
    // role separates. Lined along a contracted axis instead, it would need a value per lane off an
    // axis it does not line along, and the broadcast would silently serve the first column's value
    // to all of them.
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
