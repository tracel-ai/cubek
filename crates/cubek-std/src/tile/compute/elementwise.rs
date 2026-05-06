use cubecl::prelude::*;

use crate::tile::compute::Mask;
use crate::tile::{Plane, Tile, TileExpand};

/// Element-wise tile operations on `Tile<E, Plane, ReadWrite>`. Unlike the
/// row-wise primitives in [`crate::tile::compute::rowwise`], these touch every
/// element with no row-axis structure: a uniform scalar scale, a per-element
/// mask bool, or a whole-tile fill. Each arm delegates to a method on the
/// variant's data struct.
#[cube]
impl<E: Float> Tile<E, Plane, ReadWrite> {
    /// Multiplies each element by `scale` and adds `-inf` at masked positions.
    /// `scale` is a scalar; `mask.should_mask((r, c))` is element-wise.
    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        match self {
            Tile::Unit(t) => t.scale_and_mask::<M>(scale, mask),
            Tile::WhiteboxFragment(t) => t.scale_and_mask::<M>(scale, mask),
            Tile::Bounce(b) => b.scale_and_mask::<M>(scale, mask),
            Tile::Register(t) => t.scale_and_mask::<M>(scale, mask),
            _ => panic!("scale_and_mask: unsupported tile variant"),
        }
    }

    /// Zeros every element in the tile.
    pub fn fill_zero(&mut self) {
        match self {
            Tile::Unit(t) => t.fill_zero(),
            Tile::WhiteboxFragment(t) => t.zero(),
            Tile::Bounce(b) => b.fill_zero(),
            Tile::Cmma(t) => t.fill_zero(),
            Tile::Register(t) => t.fill_zero(),
            _ => panic!("fill_zero: unsupported tile variant"),
        }
    }
}
