use cubecl::prelude::*;

use crate::tile::{Plane, RowWise, Tile, TileExpand, TileKind, TileKindExpand};

#[cube]
impl<E: Float> Tile<E, Plane> {
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

    /// Multiply each row of `self` by `scale[r]`. The `Bounce` arm
    /// round-trips through smem to keep the cmma fragment current.
    pub fn scale_mul<SM: Float>(&mut self, scale: &RowWise<SM>) {
        let scale_e = RowWise::<SM>::cast_from::<E>(scale);
        match &mut self.kind {
            TileKind::Bounce(b) => {
                b.cmma_to_fragment();
                b.rowwise_scale(&scale_e);
                b.fragment_to_cmma();
            }
            TileKind::WhiteboxFragment(t) => t.rowwise_scale(&scale_e),
            TileKind::Unit(t) => t.rowwise_scale(&scale_e),
            TileKind::Register(t) => t.rowwise_scale(&scale_e),
            _ => panic!("scale_mul: unsupported tile variant"),
        }
    }

    /// Divide each row by `running_state_l[r]`; fully-masked rows stay zero.
    pub fn scale_div<SM: Float>(&mut self, running_state_l: &RowWise<SM>) {
        let mut scale = RowWise::<SM>::cast_from::<E>(running_state_l);
        scale.recip_inplace();
        match &mut self.kind {
            TileKind::Bounce(b) => {
                b.cmma_to_fragment();
                b.rowwise_scale(&scale);
                b.fragment_to_cmma();
            }
            TileKind::WhiteboxFragment(t) => t.rowwise_scale(&scale),
            TileKind::Unit(t) => t.rowwise_scale(&scale),
            TileKind::Register(t) => t.rowwise_scale(&scale),
            _ => panic!("scale_div: unsupported tile variant"),
        }
    }
}
