use cubecl;
use cubecl::prelude::*;

use crate::tile::data::BounceTile;

/// Internal `copy_from` between the `cmma` and `partitioned` parts of a
/// [`BounceTile`]: cmma -> smem -> partitioned. Used by the high-level
/// `softmax` / `scale_mul` / `scale_div` methods to make the partitioned view
/// current.
#[cube]
pub(crate) fn cmma_to_partitioned<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    cubecl::cmma::store(
        &mut b.smem,
        &b.cmma.matrix,
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    b.partitioned.load_from_slice(&b.smem.to_slice());
    sync_cube();
}

/// Internal `copy_from` between the `partitioned` and `cmma` parts of a
/// [`BounceTile`]: partitioned -> smem -> cmma. Reverses
/// [`cmma_to_partitioned`].
#[cube]
pub(crate) fn partitioned_to_cmma<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    b.partitioned.store_to(&mut b.smem);
    sync_cube();
    cubecl::cmma::load_with_layout(
        &b.cmma.matrix,
        &b.smem.to_slice(),
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
}
