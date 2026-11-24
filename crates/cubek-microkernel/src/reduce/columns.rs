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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::space::ComputeScope;
    use cubecl::tile::{Partitioning, Space};
    use cubek_std::devices;

    const WIDTH: usize = 4;
    const LANES: usize = 32;

    /// A sum of f32 values, its lines along `lines_along`.
    fn item(lines_along: Dimension) -> ReduceWorkItem {
        let f32 = ElemType::Float(FloatKind::F32);
        ReduceWorkItem {
            operation: ReduceOperation::Sum,
            input: f32,
            accumulator: f32,
            lines_along,
            with_indices: false,
        }
    }

    /// Lines along the reduced dimension lie one across.
    #[test]
    fn lines_along_the_row_lie_one_across() {
        let hw = devices::m2_pro();
        assert_eq!(lines_across(&hw, &item(REDUCED), WIDTH, LANES, LANES), 1);
    }

    /// Lines across it fill one memory transaction.
    #[test]
    fn lines_across_the_rows_fill_a_transaction() {
        let hw = devices::m2_pro();
        let line_bytes = WIDTH * size_of::<f32>();
        assert_eq!(
            lines_across(&hw, &item(BATCH), WIDTH, LANES, LANES),
            hw.transaction_bytes() / line_bytes
        );
    }

    /// No more of them than there are lines of rows.
    #[test]
    fn no_more_lines_than_there_are() {
        let hw = devices::m2_pro();
        assert_eq!(lines_across(&hw, &item(BATCH), WIDTH, 1, LANES), 1);
    }

    /// A power of two, where the lines of rows are not.
    #[test]
    fn as_many_lines_as_a_power_of_two() {
        let hw = devices::m2_pro();
        let lines = lines_across(&hw, &item(BATCH), 1, 3, LANES);
        assert_eq!(lines, 2);
    }

    /// A column's units take the reduced dimension in turns, a line a unit.
    #[test]
    fn a_columns_units_take_the_reduced_dimension_in_turns() {
        const COLUMNS: usize = 2;
        const PER_COLUMN: usize = 16;
        let space = Space::new(&[(BATCH, COLUMNS * WIDTH), (REDUCED, PER_COLUMN)]);
        let levels = units_across(&[(BATCH, WIDTH)], COLUMNS, PER_COLUMN).planes(&[(BATCH, 1)]);
        let partitioning = Partitioning::new(space, levels.build());
        let plane = partitioning.box_of(ComputeScope::Plane);
        assert_eq!(
            [plane.extent(BATCH), plane.extent(REDUCED)],
            [COLUMNS * WIDTH, PER_COLUMN]
        );
    }
}
