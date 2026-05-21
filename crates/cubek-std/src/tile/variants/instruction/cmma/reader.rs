use cubecl::prelude::*;

use crate::tile::StridedTile;

/// Load a CMMA fragment from a strided tile slice.
#[cube]
pub fn cmma_load_strided<E: Numeric, V: Numeric, N: Size>(
    tile: &StridedTile<V, N>,
    fragment: &mut cmma::Matrix<E>,
    layout: ComptimeOption<cmma::MatrixLayout>,
) {
    let stride = tile.unvectorized_stride();
    let slice = tile.as_slice();
    #[comptime]
    match layout {
        ComptimeOption::None => cmma::load(fragment, slice, stride),
        ComptimeOption::Some(layout) => cmma::load_with_layout(fragment, slice, stride, layout),
    }
}
