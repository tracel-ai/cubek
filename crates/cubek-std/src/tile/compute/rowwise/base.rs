use cubecl::prelude::*;

use crate::tile::{Plane, Tile, TileExpand, TileKind, TileKindExpand, data::RowWise};

/// Row-wise primitives on a `Tile<E, Plane, ReadWrite>` used for attention's
/// online softmax and output scaling. Each arm delegates to a method on the
/// variant's data struct — see `data/{unit,whitebox_fragment,bounce,register}`.
#[cube]
impl<E: Float> Tile<E, Plane, ReadWrite> {
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        match &self.kind {
            TileKind::Unit(t) => t.row_max(acc, base),
            TileKind::WhiteboxFragment(t) => t.row_max(acc, base),
            TileKind::Bounce(b) => b.row_max(acc, base),
            TileKind::Register(t) => t.row_max(acc, base),
            _ => panic!("row_max: unsupported tile variant"),
        }
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        match &self.kind {
            TileKind::Unit(t) => t.row_sum(acc),
            TileKind::WhiteboxFragment(t) => t.row_sum(acc),
            TileKind::Bounce(b) => b.row_sum(acc),
            TileKind::Register(t) => t.row_sum(acc),
            _ => panic!("row_sum: unsupported tile variant"),
        }
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        match &mut self.kind {
            TileKind::Unit(t) => t.exp_diff(rowwise),
            TileKind::WhiteboxFragment(t) => t.exp_diff(rowwise),
            TileKind::Bounce(b) => b.exp_diff(rowwise),
            TileKind::Register(t) => t.exp_diff(rowwise),
            _ => panic!("exp_diff: unsupported tile variant"),
        }
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        match &mut self.kind {
            TileKind::Unit(t) => t.rowwise_scale(scale),
            TileKind::WhiteboxFragment(t) => t.rowwise_scale(scale),
            TileKind::Bounce(b) => b.rowwise_scale(scale),
            TileKind::Register(t) => t.rowwise_scale(scale),
            _ => panic!("rowwise_scale: unsupported tile variant"),
        }
    }
}
