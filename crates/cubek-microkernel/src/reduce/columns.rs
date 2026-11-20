//! The columns a plane's units lie across where its lines run across the reduced dimension: lines
//! of rows side by side, so that one load of the plane reads whole memory transactions.

use crate::reduce::{BATCH, REDUCED, ReduceWorkItem};
use cubecl::tile::{Dimension, Levels};
use cubek_std::Hardware;

/// Lines of rows a plane of `lanes` units lies across side by side, its lines `width` wide, where
/// `row_lines` lines of rows there are: one where `item`'s lines run along the reduced dimension,
/// consecutive lanes then reading consecutive lines of one row. Where they run across it, a lane
/// reads a line of rows at one position of the reduced dimension, so a plane one line wide reads
/// part of a transaction a lane and leaves the rest to whichever plane reads the next line; as
/// many lines as one transaction holds fill it. A power of two, so a column's units are a run of
/// lane bits.
pub(crate) fn lines_across(
    hw: &Hardware,
    item: &ReduceWorkItem,
    width: usize,
    row_lines: usize,
    lanes: usize,
) -> usize {
    if item.lines_along == REDUCED {
        return 1;
    }
    let line_bytes = width * item.input.size();
    let filling = (hw.transaction_bytes() / line_bytes)
        .min(lanes)
        .min(row_lines)
        .max(1);
    1 << filling.ilog2()
}

/// The levels below a plane or a cube that lie its units across `columns` boxes of rows `owner`
/// side by side, `per_column` units in each column taking the reduced dimension in turns, one
/// line of a row each: the leaf is one position of the reduced dimension of `owner`. The reduced
/// dimension is the units' innermost, so a column's units hold the low bits of the lane.
pub(crate) fn units_across(
    owner: &[(Dimension, usize)],
    columns: usize,
    per_column: usize,
) -> Levels {
    let mut leaf = owner.to_vec();
    leaf.push((REDUCED, 1));
    Levels::leaf(&leaf).units(&[(BATCH, columns), (REDUCED, per_column)])
}
