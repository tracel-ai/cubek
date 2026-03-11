use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::ReduceOp;
use crate::components::softmax::Reducer;
use crate::components::softmax::base::SoftmaxConfig;
use crate::components::tile::RowWise;
use crate::components::tile::SoftmaxRowwise;

#[derive(CubeType)]
/// Trivial reducer for one unit
pub struct UnitReducer {}

#[cube]
impl Reducer for UnitReducer {
    fn reduce<E: Float, F: SoftmaxRowwise<E>, RO: ReduceOp<E>>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] _config: SoftmaxConfig,
    ) {
        RO::reduce_local_accumulate::<F>(data, vals);
    }
}
