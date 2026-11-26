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
fn units_across(owner: &[(Dimension, usize)], columns: usize, per_column: usize) -> Levels {
    let mut leaf = owner.to_vec();
    leaf.push((REDUCED, 1));
    Levels::leaf(&leaf).units(&[(BATCH, columns), (REDUCED, per_column)])
}

/// Whether `lanes` units lie across `columns` lines of rows, a run of lane bits a column.
pub(crate) const fn spreads_across(lanes: usize, columns: usize) -> bool {
    columns.is_power_of_two() && columns <= lanes
}

/// The levels of a plane of `lanes` units that lie across `lines` boxes of rows `owner` side by side:
/// the one box, read flat, or a column of units a box ([`plane_columns`]).
pub(crate) fn plane_levels(owner: &[(Dimension, usize)], lines: usize, lanes: usize) -> Levels {
    match lines {
        1 => Levels::leaf(owner),
        columns => plane_columns(owner, columns, lanes),
    }
}

/// The levels of a plane of `lanes` units that lie across `columns` boxes of rows `owner`, every step
/// along the reduced dimension the plane's to walk.
pub(crate) fn plane_columns(owner: &[(Dimension, usize)], columns: usize, lanes: usize) -> Levels {
    units_across(owner, columns, lanes / columns).walk_every(&[REDUCED])
}

/// The levels of a cube of `planes` planes of `lanes` units that lie across `lines` boxes of rows
/// `owner` side by side: the one box, read flat by all of its units, or a column of each plane's units
/// a box ([`cube_columns`]).
pub(crate) fn cube_levels(
    owner: &[(Dimension, usize)],
    lines: usize,
    lanes: usize,
    planes: usize,
) -> Levels {
    match lines {
        1 => Levels::leaf(owner),
        columns => cube_columns(owner, columns, lanes, planes),
    }
}

/// [`plane_columns`] of each of `planes` planes of `lanes` units, the planes after one another along
/// the reduced dimension.
pub(crate) fn cube_columns(
    owner: &[(Dimension, usize)],
    columns: usize,
    lanes: usize,
    planes: usize,
) -> Levels {
    units_across(owner, columns, lanes / columns)
        .planes(&[(REDUCED, planes)])
        .walk_every(&[REDUCED])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::space::ComputeScope;
    use cubecl::tile::{Partitioning, Space};
    use cubek_std::devices;

    const LINE_LANE_COUNT: usize = 4;
    const LANE_COUNT: usize = 32;

    #[test]
    fn a_column_a_lane_spreads_across() {
        assert!(spreads_across(LANE_COUNT, LANE_COUNT));
    }

    #[test]
    fn more_columns_than_lanes_do_not() {
        assert!(!spreads_across(LANE_COUNT, 2 * LANE_COUNT));
    }

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

    #[test]
    fn lines_along_the_row_lie_one_across() {
        let hw = devices::m2_pro();
        assert_eq!(
            lines_across(&hw, &item(REDUCED), LINE_LANE_COUNT, LANE_COUNT, LANE_COUNT),
            1
        );
    }

    /// Lines across it fill one memory transaction.
    #[test]
    fn lines_across_the_rows_fill_a_transaction() {
        let hw = devices::m2_pro();
        let line_bytes = LINE_LANE_COUNT * size_of::<f32>();
        assert_eq!(
            lines_across(&hw, &item(BATCH), LINE_LANE_COUNT, LANE_COUNT, LANE_COUNT),
            hw.transaction_bytes() / line_bytes
        );
    }

    #[test]
    fn no_more_lines_than_there_are() {
        let hw = devices::m2_pro();
        assert_eq!(
            lines_across(&hw, &item(BATCH), LINE_LANE_COUNT, 1, LANE_COUNT),
            1
        );
    }

    /// A power of two, where the lines of rows are not.
    #[test]
    fn as_many_lines_as_a_power_of_two() {
        let hw = devices::m2_pro();
        let lines = lines_across(&hw, &item(BATCH), 1, 3, LANE_COUNT);
        assert_eq!(lines, 2);
    }

    #[test]
    fn a_columns_units_take_the_reduced_dimension_in_turns() {
        const COLUMN_COUNT: usize = 2;
        const COLUMN_LANE_COUNT: usize = 16;
        let space = Space::new(&[
            (BATCH, COLUMN_COUNT * LINE_LANE_COUNT),
            (REDUCED, COLUMN_LANE_COUNT),
        ]);
        let levels = units_across(&[(BATCH, LINE_LANE_COUNT)], COLUMN_COUNT, COLUMN_LANE_COUNT)
            .planes(&[(BATCH, 1)]);
        let partitioning = Partitioning::new(space, levels.build());
        let plane = partitioning.box_of(ComputeScope::Plane);
        assert_eq!(
            [plane.extent(BATCH), plane.extent(REDUCED)],
            [COLUMN_COUNT * LINE_LANE_COUNT, COLUMN_LANE_COUNT]
        );
    }

    /// One line wide, the plane's box of rows is the leaf: no level cuts it to units.
    #[test]
    fn one_line_across_is_the_owners_box() {
        let levels = plane_levels(&[(BATCH, LINE_LANE_COUNT)], 1, LANE_COUNT).build();
        assert_eq!(levels, Levels::leaf(&[(BATCH, LINE_LANE_COUNT)]).build());
    }

    /// A cube's planes take the reduced dimension after one another.
    #[test]
    fn a_cubes_planes_follow_one_another_along_the_reduced_dimension() {
        const COLUMN_COUNT: usize = 2;
        const PLANE_COUNT: usize = 4;
        let positions = LANE_COUNT / COLUMN_COUNT * PLANE_COUNT;
        let space = Space::new(&[
            (BATCH, COLUMN_COUNT * LINE_LANE_COUNT),
            (REDUCED, positions),
        ]);
        let levels = cube_columns(
            &[(BATCH, LINE_LANE_COUNT)],
            COLUMN_COUNT,
            LANE_COUNT,
            PLANE_COUNT,
        )
        .cubes(&[BATCH]);
        let partitioning = Partitioning::new(space, levels.build());
        let cube = partitioning.box_of(ComputeScope::Cube);
        assert_eq!(cube.extent(REDUCED), positions);
    }

    /// One line wide, a cube's box of rows is the leaf too.
    #[test]
    fn one_line_across_a_cube_is_the_owners_box() {
        const PLANE_COUNT: usize = 4;
        let levels = cube_levels(&[(BATCH, LINE_LANE_COUNT)], 1, LANE_COUNT, PLANE_COUNT).build();
        assert_eq!(levels, Levels::leaf(&[(BATCH, LINE_LANE_COUNT)]).build());
    }

    /// Several lines wide, a plane's levels are its columns.
    #[test]
    fn several_lines_across_are_columns() {
        const COLUMN_COUNT: usize = 2;
        let owner = [(BATCH, LINE_LANE_COUNT)];
        let levels = plane_levels(&owner, COLUMN_COUNT, LANE_COUNT).build();
        assert_eq!(
            levels,
            plane_columns(&owner, COLUMN_COUNT, LANE_COUNT).build()
        );
    }

    /// Several lines wide, a cube's levels are its columns.
    #[test]
    fn several_lines_across_a_cube_are_columns() {
        const COLUMN_COUNT: usize = 2;
        const PLANE_COUNT: usize = 4;
        let owner = [(BATCH, LINE_LANE_COUNT)];
        let levels = cube_levels(&owner, COLUMN_COUNT, LANE_COUNT, PLANE_COUNT).build();
        let columns = cube_columns(&owner, COLUMN_COUNT, LANE_COUNT, PLANE_COUNT).build();
        assert_eq!(levels, columns);
    }
}
