use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::pipeline::{InnerLayout, LocalTile, LocalTileLayout};

/// Comptime configuration for constructing a [`RowwiseTileWorkspace`].
///
/// `Direct` is for fragments where each unit holds its own copy of the tile
/// (e.g. register fragments) — no smem bounce is needed.
/// `Bounce` is for opaque fragments (e.g. cmma) where the row-wise operations
/// need to round-trip through shared memory and a `LocalTile` view.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RowwiseTileKind {
    Direct,
    Bounce(BounceConfig),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BounceConfig {
    pub tile_shape: Coords2d,
    pub num_planes: u32,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

#[derive(CubeType)]
pub enum RowwiseTileWorkspace<E: Numeric> {
    Direct,
    Bounce(BounceWorkspace<E>),
}

#[derive(CubeType)]
pub struct BounceWorkspace<E: Numeric> {
    pub smem: SliceMut<E>,
    pub local_tile: LocalTile<E>,
}

#[cube]
impl<E: Numeric> RowwiseTileWorkspace<E> {
    pub fn new(#[comptime] kind: RowwiseTileKind) -> RowwiseTileWorkspace<E> {
        match kind {
            RowwiseTileKind::Direct => RowwiseTileWorkspace::new_Direct(),
            RowwiseTileKind::Bounce(cfg) => {
                let total_tile_size = (cfg.tile_shape.0 * cfg.tile_shape.1) as usize;
                let smem_size = total_tile_size * cfg.num_planes as usize;
                let start = UNIT_POS_Y as usize * total_tile_size;
                let end = start + total_tile_size;
                let smem = SharedMemory::new(smem_size).slice_mut(start, end);

                let layout = comptime! {
                    LocalTileLayout::new(cfg.tile_shape, cfg.plane_dim, cfg.inner_layout)
                };
                let local_tile = LocalTile::new(layout);

                RowwiseTileWorkspace::new_Bounce(BounceWorkspace::<E> { smem, local_tile })
            }
        }
    }
}
