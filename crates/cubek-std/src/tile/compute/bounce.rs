use cubecl;
use cubecl::prelude::*;

use crate::tile::data::BounceTile;

/// Internal `copy_from` between the `cmma` and `fragment` parts of a
/// [`BounceTile`]: cmma -> smem -> fragment. Used by the high-level
/// `softmax` / `scale_mul` / `scale_div` methods to make the fragment view
/// current.
#[cube]
pub(crate) fn cmma_to_whitebox_fragment<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    cubecl::cmma::store(
        &mut b.smem,
        &b.cmma.matrix,
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    b.fragment.load_from_slice(&b.smem.to_slice());
    sync_cube();
}

/// Internal `copy_from` between the `fragment` and `cmma` parts of a
/// [`BounceTile`]: fragment -> smem -> cmma. Reverses
/// [`cmma_to_whitebox_fragment`].
#[cube]
pub(crate) fn whitebox_fragment_to_cmma<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    b.fragment.store_to(&mut b.smem);
    sync_cube();
    cubecl::cmma::load_with_layout(
        &b.cmma.matrix,
        &b.smem.to_slice(),
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
}
