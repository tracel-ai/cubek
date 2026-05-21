//! `Tile::should_mask` and `Tile::load_mask_from_strided_tile` dispatchers.

use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::{
    StridedTile, Tile, TileExpand, TileKind, TileKindExpand, TileScope,
    mask::{Mask, MaskExpand},
};

#[cube]
impl<E: Numeric, Sc: TileScope> Mask for Tile<E, Sc> {
    fn should_mask(&self, local_pos: Coords2d) -> bool {
        match &self.kind {
            TileKind::Unit(t) => t.should_mask(local_pos),
            TileKind::WhiteboxFragment(t) => t.should_mask(local_pos),
            _ => panic!(
                "Mask::should_mask is only defined for Tile::Unit and Tile::WhiteboxFragment"
            ),
        }
    }
}

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc> {
    /// Materialize a mask fragment from a `StridedTile` into `Unit` or
    /// `WhiteboxFragment`.
    pub fn load_mask_from_strided_tile<E: Numeric, ES: Size>(&mut self, tile: &StridedTile<E, ES>) {
        match &mut self.kind {
            TileKind::Unit(t) => t.load_from_strided_tile::<E, ES>(tile),
            TileKind::WhiteboxFragment(t) => t.load_from_strided_tile::<E, ES>(tile),
            _ => panic!("load_mask_from_strided_tile: unsupported tile variant"),
        }
    }
}
