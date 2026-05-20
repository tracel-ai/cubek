use cubecl::prelude::*;

use crate::tile::mask::Mask;
use crate::tile::{Plane, Tile, TileExpand, TileKind, TileKindExpand};

#[cube]
impl<E: Float> Tile<E, Plane> {
    /// Multiply by `scale` and add `-inf` at masked positions.
    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        match &mut self.kind {
            TileKind::Unit(t) => t.scale_and_mask::<M>(scale, mask),
            TileKind::WhiteboxFragment(t) => t.scale_and_mask::<M>(scale, mask),
            TileKind::Bounce(b) => b.scale_and_mask::<M>(scale, mask),
            TileKind::Register(t) => t.scale_and_mask::<M>(scale, mask),
            _ => panic!("scale_and_mask: unsupported tile variant"),
        }
    }

    pub fn fill_zero(&mut self) {
        match &mut self.kind {
            TileKind::Unit(t) => t.fill_zero(),
            TileKind::WhiteboxFragment(t) => t.zero(),
            TileKind::Bounce(b) => b.fill_zero(),
            TileKind::Cmma(t) => t.fill_zero(),
            TileKind::Register(t) => t.fill_zero(),
            TileKind::RowWise(t) => t.init_zero(),
            _ => panic!("fill_zero: unsupported tile variant"),
        }
    }
}
