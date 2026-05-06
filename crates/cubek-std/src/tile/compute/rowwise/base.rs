use cubecl::prelude::*;

use crate::tile::{Plane, Tile, TileExpand, data::RowWise};

/// Row-wise primitives on a `Tile<E, Plane, ReadWrite>` used for attention's
/// online softmax and output scaling. Each arm delegates to a method on the
/// variant's data struct — see `data/{unit,whitebox_fragment,bounce,register}`.
#[cube]
impl<E: Float> Tile<E, Plane, ReadWrite> {
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        match self {
            Tile::Unit(t) => t.row_max(acc, base),
            Tile::WhiteboxFragment(t) => t.row_max(acc, base),
            Tile::Bounce(b) => b.row_max(acc, base),
            Tile::Register(t) => t.row_max(acc, base),
            _ => panic!("row_max: unsupported tile variant"),
        }
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        match self {
            Tile::Unit(t) => t.row_sum(acc),
            Tile::WhiteboxFragment(t) => t.row_sum(acc),
            Tile::Bounce(b) => b.row_sum(acc),
            Tile::Register(t) => t.row_sum(acc),
            _ => panic!("row_sum: unsupported tile variant"),
        }
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        match self {
            Tile::Unit(t) => t.exp_diff(rowwise),
            Tile::WhiteboxFragment(t) => t.exp_diff(rowwise),
            Tile::Bounce(b) => b.exp_diff(rowwise),
            Tile::Register(t) => t.exp_diff(rowwise),
            _ => panic!("exp_diff: unsupported tile variant"),
        }
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        match self {
            Tile::Unit(t) => t.rowwise_scale(scale),
            Tile::WhiteboxFragment(t) => t.rowwise_scale(scale),
            Tile::Bounce(b) => b.rowwise_scale(scale),
            Tile::Register(t) => t.rowwise_scale(scale),
            _ => panic!("rowwise_scale: unsupported tile variant"),
        }
    }
}
