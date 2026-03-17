use cubecl;
use cubecl::prelude::*;

use crate::components::tile::pipeline::RowWise;

#[cube]
/// Strategy for reducing across units participating in the same row
pub trait Reducer<E: Float>: CubeType {
    type Tile: CubeType;

    /// Computes the sum of rows on a fragment, using the Reducer's strategy
    fn row_sum(vals: &mut RowWise<E>, data: &Self::Tile);

    /// Computes the max of rows on a fragment, using the Reducer's strategy
    /// Starts max at base
    fn row_max(vals: &mut RowWise<E>, base: &RowWise<E>, data: &Self::Tile);
}
