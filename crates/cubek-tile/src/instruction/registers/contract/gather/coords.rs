//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::*;

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
    #[comptime] acc: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] width: usize,
) -> Vector<T, W> {
    view.read(cell_position(
        batch,
        row,
        col,
        reduce_coords,
        operand,
        acc,
        reduce,
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
    #[comptime] acc: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] width: usize,
) -> CoordsDyn {
    let acc_coords = acc_cell_coords(batch, row, col);
    resolve_nd_coords(
        operand,
        acc,
        reduce,
        &acc_coords,
        reduce_coords,
        width,
        false,
    )
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

/// What [`resolve_nd_coords`] and the lane fold assume about how the operands are lined up. Both
/// treat one axis per operand as the vectorized one and address it in lines; if that is not the
/// axis the operand actually lines along, the reads are silently off by the width rather than
/// wrong in a way a test would localize. Host-side, so a violation is a comptime message.
pub(super) fn assert_operand_shapes(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    reduce: &[Axis],
    rhs_vec_len: usize,
    lhs_spans_col: bool,
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
    assert!(
        lhs.axis_at(lhs.rank() - 1) == fastest,
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
    // An lhs varying along the column is read once per cell, and a cell is `rhs_vec_len` columns
    // wide. The lhs lines along a contracted axis, not this one, so one read cannot cover them:
    // it would need a value per lane off an axis it does not line along, and the broadcast would
    // silently serve the first column's value to all of them.
    assert!(
        !lhs_spans_col || rhs_vec_len == 1,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis cannot also be served in lines (the accumulator is {rhs_vec_len} \
         wide)"
    );
}
