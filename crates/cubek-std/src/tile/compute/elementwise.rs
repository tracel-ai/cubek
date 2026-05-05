use cubecl;
use cubecl::prelude::*;

use crate::tile::compute::{Mask, MaskExpand};
use crate::tile::{Plane, Tile, TileExpand};

/// Element-wise tile operations on `Tile<E, Plane, ReadWrite>`. Unlike the
/// row-wise primitives in [`crate::tile::compute::rowwise`], these touch every
/// element with no row-axis structure: a uniform scalar scale, a per-element
/// mask bool, or a whole-tile fill.
#[cube]
impl<E: Float> Tile<E, Plane, ReadWrite> {
    /// Multiplies each element by `scale` and adds `-inf` at masked positions.
    /// `scale` is a scalar; `mask.should_mask((r, c))` is element-wise.
    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        match self {
            Tile::Unit(t) => t.scale_and_mask::<M>(scale, mask),
            Tile::WhiteboxFragment(t) => t.scale_and_mask::<M>(scale, mask),
            Tile::Bounce(b) => b.fragment.scale_and_mask::<M>(scale, mask),
            Tile::Register(t) => {
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                for r in 0..m {
                    let row_offset = r * n;
                    for c in 0..n {
                        let idx = (row_offset + c) as usize;
                        t.data[idx] = t.data[idx] * scale
                            + E::cast_from(mask.should_mask((r, c))) * E::min_value();
                    }
                }
            }
            _ => panic!("scale_and_mask: unsupported tile variant"),
        }
    }

    /// Zeros every element in the tile.
    pub fn fill_zero(&mut self) {
        match self {
            Tile::Register(t) => {
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                for i in 0..m * n {
                    t.data[i as usize] = E::from_int(0);
                }
            }
            Tile::Unit(t) => t.zero(),
            Tile::WhiteboxFragment(t) => t.zero(),
            Tile::Bounce(b) => {
                cubecl::cmma::fill(&b.cmma.matrix, E::from_int(0));
            }
            Tile::Cmma(t) => {
                cubecl::cmma::fill(&t.matrix, E::from_int(0));
            }
            _ => panic!("fill_zero: unsupported tile variant"),
        }
    }
}
