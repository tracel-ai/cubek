use cubecl::prelude::*;

use crate::tile::mask::Mask;
use crate::tile::{Plane, Tile, TileExpand, TileKind, TileKindExpand};

/// Element-wise tile operations on `Tile<E, Plane, ReadWrite>`. Unlike the
/// row-wise primitives in [`crate::tile::ops::rowwise`], these touch every
/// element with no row-axis structure: a uniform scalar scale, a per-element
/// mask bool, or a whole-tile fill. Each arm delegates to a method on the
/// variant's data struct.
#[cube]
impl<E: Float> Tile<E, Plane, ReadWrite> {
    /// Multiplies each element by `scale` and adds `-inf` at masked positions.
    /// `scale` is a scalar; `mask.should_mask((r, c))` is element-wise.
    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        match &mut self.kind {
            TileKind::Unit(t) => t.scale_and_mask::<M>(scale, mask),
            TileKind::WhiteboxFragment(t) => t.scale_and_mask::<M>(scale, mask),
            TileKind::Bounce(b) => b.scale_and_mask::<M>(scale, mask),
            TileKind::Register(t) => t.scale_and_mask::<M>(scale, mask),
            _ => panic!("scale_and_mask: unsupported tile variant"),
        }
    }

    /// Zeros every element in the tile.
    pub fn fill_zero(&mut self) {
        match &mut self.kind {
            TileKind::Unit(t) => t.fill_zero(),
            TileKind::WhiteboxFragment(t) => t.zero(),
            TileKind::Bounce(b) => b.fill_zero(),
            TileKind::Cmma(t) => t.fill_zero(),
            TileKind::Register(t) => t.fill_zero(),
            _ => panic!("fill_zero: unsupported tile variant"),
        }
    }
}
