use cubecl;
use cubecl::prelude::*;

use crate::components::tile::{
    pipeline::{RowWise, UnitTile},
    softmax::Reducer,
};

#[derive(CubeType)]
/// Trivial reducer for one unit
pub struct UnitReducer {}

#[cube]
impl<E: Float> Reducer<E> for UnitReducer {
    type Tile = UnitTile<E>;

    fn row_sum(vals: &mut RowWise<E>, data: &Self::Tile) {
        vals.fill(E::from_int(0));
        vals.add_inplace(&data.rowwise_sum())
    }

    fn row_max(vals: &mut RowWise<E>, base: &RowWise<E>, data: &Self::Tile) {
        vals.copy_from(base);
        vals.max_inplace(&data.rowwise_max())
    }
}
