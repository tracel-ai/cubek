use cubecl;
use cubecl::prelude::*;

use crate::tile::scope::{Scope, assert_plane_scope};
use crate::tile::variants::{InnerLayout, LocalTile, LocalTileLayout};
use crate::tile::{CmmaTile, Tile};

/// Comptime configuration for [`BounceTile`].
///
/// A bounce tile bundles an opaque cmma fragment together with a shared-memory
/// scratch slice and a [`LocalTile`] view, so row-wise operations can be
/// expressed as `copy_from` between the inner pieces. From the caller's point
/// of view it is a single [`Tile`] variant — only valid when the tile's
/// scope is `Plane`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BounceConfig {
    pub tile_shape: (u32, u32),
    pub num_planes: u32,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

#[derive(CubeType)]
pub struct BounceTile<N: Numeric> {
    pub cmma: CmmaTile<N>,
    pub smem: SliceMut<N>,
    pub local: LocalTile<N>,
}

#[cube]
impl<N: Numeric> BounceTile<N> {
    pub fn new(cmma: CmmaTile<N>, #[comptime] cfg: BounceConfig) -> BounceTile<N> {
        let total_tile_size = comptime!((cfg.tile_shape.0 * cfg.tile_shape.1) as usize);
        let smem_size = comptime!(total_tile_size * cfg.num_planes as usize);
        let start = UNIT_POS_Y as usize * total_tile_size;
        let end = start + total_tile_size;
        let smem = SharedMemory::new(smem_size).slice_mut(start, end);

        let layout = comptime!(LocalTileLayout::new(
            cfg.tile_shape,
            cfg.plane_dim,
            cfg.inner_layout
        ));
        let local = LocalTile::new(layout);

        BounceTile::<N> { cmma, smem, local }
    }
}

#[cube]
/// Wraps a freshly built `CmmaTile` in a `Tile::Bounce`. Panics at expansion
/// time unless `Sc = Plane`.
pub fn allocate_bounce_tile<E: Numeric, Sc: Scope>(
    cmma: CmmaTile<E>,
    #[comptime] cfg: BounceConfig,
) -> Tile<E, Sc, ReadWrite> {
    comptime!(assert_plane_scope(Sc::KIND));
    Tile::new_Bounce(BounceTile::<E>::new(cmma, cfg))
}
