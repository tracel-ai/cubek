//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::*;

/// Read one operand at an accumulator cell and a point in the contracted space.
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

/// Assemble the accumulator coordinate expected by [`resolve_nd_coords`].
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

/// Validate the coordinate and vectorization assumptions made by both gather schedules.
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
    assert!(
        rhs_vec_len == 1 || rhs.axis_at(rhs.rank() - 1) == acc.axis_at(acc.rank() - 1),
        "contract gather: a vectorized rhs must line along the accumulator's innermost axis"
    );
    assert!(
        !lhs_spans_col || rhs_vec_len == 1,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis cannot also be served in lines (the accumulator is {rhs_vec_len} \
         wide)"
    );
}
