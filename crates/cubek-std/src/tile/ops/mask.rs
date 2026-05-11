//! `Tile::should_mask` (via the `Mask` trait) and `Tile::load_mask_from_strided_tile`
//! dispatchers. Each arm delegates to a method on the variant's data struct
//! in [`crate::tile::variants`]. The `Mask` trait, `MaskLayout` enum, and the
//! mask-layout helpers live at [`crate::tile::mask`] (top-level support
//! types, not dispatchers).

use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::{
    Tile, TileExpand, TileKind, TileKindExpand, TileScope,
    mask::{Mask, MaskExpand},
    variants::StridedTile,
};

#[cube]
impl<E: Numeric, Sc: TileScope, IO: SliceVisibility> Mask for Tile<E, Sc, IO> {
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
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Loads the data from an external strided tile into the inner storage of a
    /// `Tile::Unit` or `Tile::WhiteboxFragment`. Used to materialize a mask fragment.
    pub fn load_mask_from_strided_tile<E: Numeric, ES: Size>(&mut self, tile: &StridedTile<E, ES>) {
        match &mut self.kind {
            TileKind::Unit(t) => t.load_from_strided_tile::<E, ES>(tile),
            TileKind::WhiteboxFragment(t) => t.load_from_strided_tile::<E, ES>(tile),
            _ => panic!("load_mask_from_strided_tile: unsupported tile variant"),
        }
    }
}
