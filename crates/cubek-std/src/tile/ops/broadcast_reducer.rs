use cubecl;
use cubecl::prelude::*;

use crate::tile::ops::RowWise;
use crate::tile::variants::LocalTile;

/// Reduces row-wise quantities across units of a plane that participate in the
/// same row, masking out off-row peers. Used internally by the row-wise
/// operations on `Tile<E, Plane, ReadWrite>` when dispatching to the
/// `Tile::Local` (and `Tile::Bounce`) arms.
///
/// Restricted to plane scope by virtue of using `plane_shuffle` and
/// `UNIT_POS_X` — callers are expected to enforce that constraint.
#[cube]
pub(crate) fn local_row_max<E: Float>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &LocalTile<E>,
) {
    vals.copy_from(base);
    reduce::<E, LocalRowMax>(vals, data)
}

#[cube]
pub(crate) fn local_row_sum<E: Float>(vals: &mut RowWise<E>, data: &LocalTile<E>) {
    vals.fill(E::from_int(0));
    reduce::<E, LocalRowSum>(vals, data)
}

#[cube]
fn reduce<E: Float, RO: ReduceOp<E>>(vals: &mut RowWise<E>, data: &LocalTile<E>) {
    let num_units_per_row = data.num_units_per_row().comptime();
    let num_shares_within_plane = num_units_per_row.next_power_of_two().ilog2();

    let unit_pos = UNIT_POS_X;
    let unit_pos_in_row = unit_pos % num_units_per_row;

    RO::reduce_local(data, vals);

    for i in 0..num_shares_within_plane {
        let offset = num_units_per_row >> (i + 1);
        let source_unit = unit_pos + offset;

        let value_from_source = rowwise_plane_broadcast(vals, source_unit);

        // Mask if outside the row
        let mask = unit_pos_in_row + offset >= num_units_per_row;
        RO::reduce_from_peer(vals, &value_from_source, mask);
    }

    // Broadcast back to subgroup
    let result = &rowwise_plane_broadcast(vals, unit_pos - unit_pos_in_row);
    vals.copy_from(result);
}

#[cube]
fn rowwise_plane_broadcast<E: Float>(rowwise: &RowWise<E>, source_unit: u32) -> RowWise<E> {
    let mut result = Array::new(rowwise.num_rows);

    for r in 0..rowwise.num_rows {
        result[r] = plane_shuffle(rowwise.vals[r], source_unit);
    }

    RowWise::<E> {
        num_rows: rowwise.num_rows,
        vals: result,
    }
}

#[cube]
trait ReduceOp<E: Float> {
    fn reduce_local(data: &LocalTile<E>, acc: &mut RowWise<E>);
    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}

#[derive(CubeType)]
struct LocalRowMax {}

#[derive(CubeType)]
struct LocalRowSum {}

#[cube]
impl<E: Float> ReduceOp<E> for LocalRowMax {
    fn reduce_local(data: &LocalTile<E>, acc: &mut RowWise<E>) {
        acc.max_inplace(&data.rowwise_max())
    }

    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(mask) * E::min_value());
        masked.add_inplace(elem);

        acc.max_inplace(&masked)
    }
}

#[cube]
impl<E: Float> ReduceOp<E> for LocalRowSum {
    fn reduce_local(data: &LocalTile<E>, acc: &mut RowWise<E>) {
        acc.add_inplace(&data.rowwise_sum())
    }

    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(!mask));
        masked.mul_inplace(elem);

        acc.add_inplace(&masked)
    }
}
