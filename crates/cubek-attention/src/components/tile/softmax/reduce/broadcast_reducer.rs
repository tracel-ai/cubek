use cubecl;
use cubecl::prelude::*;

use crate::components::tile::{
    pipeline::{LocalTile, RowVal, RowWise},
    softmax::Reducer,
};

#[derive(CubeType)]
/// Applies reduction on rows, masking planes that do not participate in the row
pub struct BroadcastReducer {}

#[cube]
impl<E: Float> Reducer<E> for BroadcastReducer {
    type Tile = LocalTile<E>;

    fn row_sum(vals: &mut RowWise<E>, data: &Self::Tile) {
        vals.fill(E::from_int(0));
        reduce::<E, LocalRowSum>(vals, data)
    }

    fn row_max(vals: &mut RowWise<E>, base: &RowWise<E>, data: &Self::Tile) {
        vals.copy_from(base);
        reduce::<E, LocalRowMax>(vals, data)
    }
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
fn rowwise_plane_broadcast<E: Float>(val: &RowWise<E>, source_unit: u32) -> RowWise<E> {
    let mut result = Sequence::new();

    #[unroll]
    for row in 0..val.num_rows {
        result.push(RowVal::<E> {
            val: plane_shuffle(val.index(row), source_unit),
        });
    }

    RowWise::<E> {
        num_rows: val.num_rows,
        vals: result,
    }
}

#[cube]
/// A reduction operation
pub trait ReduceOp<E: Float> {
    /// Applies the reduction on the elements of the same row held by the unit,
    /// and store in the accumulator
    fn reduce_local(data: &LocalTile<E>, acc: &mut RowWise<E>);

    /// Accumulates elem into acc.
    /// If mask is activated, the element gets masked prior to being accumulated
    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}

#[derive(CubeType)]
/// Max reduction operation
pub struct LocalRowMax {}

#[derive(CubeType)]
/// Sum reduction operation
pub struct LocalRowSum {}

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
