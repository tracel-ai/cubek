//! The RMS norm deferred into the products around it: the first product reduces each row's squares
//! into the norm's factor as it stores ([`RmsNormFactor`]), and the second scales each row by that
//! factor as it stores ([`RowScale`](cubecl::tile::launch::RowScale)).

use cubecl::prelude::*;
use cubecl::tile::launch::RowReduction;
use cubek_microkernel::rms_norm::RmsNormScale;

/// A row's RMS norm factor: its squares summed, then scaled as [`RmsNormScale`] says.
pub struct RmsNormFactor;

#[cube]
impl RowReduction for RmsNormFactor {
    type Param = RmsNormScale;

    fn cells<E: Numeric, N: Size>(value: Vector<E, N>) -> Vector<f32, N> {
        let wide = Vector::<f32, N>::cast_from(value);
        wide * wide
    }

    fn of_sum(sum: f32, param: &RmsNormScale) -> f32 {
        param.of(sum)
    }
}
