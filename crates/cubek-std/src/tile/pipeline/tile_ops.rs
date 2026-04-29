use cubecl;
use cubecl::prelude::*;

use crate::tile::pipeline::{
    BroadcastReducer, LOGIT_MASKED, Mask, MaskExpand, RowWise, RowwiseTileWorkspace,
    RowwiseTileWorkspaceExpand,
};
use crate::tile::{Plane, Tile, TileExpand};

/// Row-wise primitives on a `Tile<E, V, Plane, ReadWrite>` used for attention's
/// softmax and output scaling. The caller drives the lifecycle:
///
/// 1. Optional `bounce_in(ws, stride)` — only relevant for `Tile::Cmma`, which
///    stores its fragment to shared memory and loads it into the workspace's
///    `local_tile` so subsequent ops can run on a unit-addressable view.
///    For `Tile::Register` this is a no-op since the data is already directly
///    accessible.
/// 2. Run row-wise ops (`row_max`, `row_sum`, `exp_diff`, `rowwise_scale`,
///    `scale_and_mask`). For `Cmma` they operate on `ws.local_tile`; for
///    `Register` they operate on the tile's own array.
/// 3. Optional `bounce_out(ws, stride)` — for `Cmma`, writes `ws.local_tile`
///    back to the cmma fragment via shared memory. No-op for `Register`.
#[cube]
impl<E: Float, V: Size> Tile<E, V, Plane, ReadWrite> {
    pub fn bounce_in(&self, ws: &mut RowwiseTileWorkspace<E>) {
        match (self, ws) {
            (Tile::Register(_), _) => {}
            (Tile::Cmma(t), RowwiseTileWorkspace::Bounce(bw)) => {
                let stride = comptime!(t.tile_size.n());
                cmma::store(
                    &mut bw.smem,
                    &t.matrix,
                    stride,
                    cmma::MatrixLayout::RowMajor,
                );
                sync_cube();
                bw.local_tile.load_from_slice(&bw.smem.to_slice());
                sync_cube();
            }
            _ => panic!("bounce_in: variant mismatch between tile and workspace"),
        }
    }

    pub fn bounce_out(&mut self, ws: &mut RowwiseTileWorkspace<E>) {
        match (self, ws) {
            (Tile::Register(_), _) => {}
            (Tile::Cmma(t), RowwiseTileWorkspace::Bounce(bw)) => {
                let stride = comptime!(t.tile_size.n());
                bw.local_tile.store_to(&mut bw.smem);
                sync_cube();
                cmma::load_with_layout(
                    &t.matrix,
                    &bw.smem.to_slice(),
                    stride,
                    cmma::MatrixLayout::RowMajor,
                );
            }
            _ => panic!("bounce_out: variant mismatch"),
        }
    }

    pub fn row_max(
        &self,
        acc: &mut RowWise<E>,
        base: &RowWise<E>,
        ws: &RowwiseTileWorkspace<E>,
    ) {
        match (self, ws) {
            (Tile::Register(t), _) => {
                acc.copy_from(base);
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                for r in 0..m as usize {
                    let row_offset = r as u32 * n;
                    let mut val = E::min_value();
                    for c in 0..n {
                        val = max(val, t.data[(row_offset + c) as usize]);
                    }
                    acc.vals[r] = max(acc.vals[r], val);
                }
            }
            (Tile::Cmma(_), RowwiseTileWorkspace::Bounce(bw)) => {
                BroadcastReducer::row_max::<E>(acc, base, &bw.local_tile);
            }
            _ => panic!("row_max: variant mismatch"),
        }
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>, ws: &RowwiseTileWorkspace<E>) {
        match (self, ws) {
            (Tile::Register(t), _) => {
                acc.fill(E::from_int(0));
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                for r in 0..m as usize {
                    let row_offset = r as u32 * n;
                    let mut val = E::from_int(0);
                    for c in 0..n {
                        val += t.data[(row_offset + c) as usize];
                    }
                    acc.vals[r] += val;
                }
            }
            (Tile::Cmma(_), RowwiseTileWorkspace::Bounce(bw)) => {
                BroadcastReducer::row_sum::<E>(acc, &bw.local_tile);
            }
            _ => panic!("row_sum: variant mismatch"),
        }
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>, ws: &mut RowwiseTileWorkspace<E>) {
        match (self, ws) {
            (Tile::Register(t), _) => {
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                let threshold = E::new(LOGIT_MASKED);
                for r in 0..m as usize {
                    let row_offset = r as u32 * n;
                    let val = rowwise.vals[r];
                    let safe_val = clamp_min(val, threshold);
                    let not_masked = E::cast_from(val >= threshold);
                    for c in 0..n {
                        let idx = (row_offset + c) as usize;
                        t.data[idx] = not_masked * (t.data[idx] - safe_val).exp();
                    }
                }
            }
            (Tile::Cmma(_), RowwiseTileWorkspace::Bounce(bw)) => {
                bw.local_tile.exp_diff(rowwise);
            }
            _ => panic!("exp_diff: variant mismatch"),
        }
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>, ws: &mut RowwiseTileWorkspace<E>) {
        match (self, ws) {
            (Tile::Register(t), _) => {
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                for r in 0..m as usize {
                    let row_offset = r as u32 * n;
                    for c in 0..n {
                        let idx = (row_offset + c) as usize;
                        t.data[idx] = t.data[idx] * scale.vals[r];
                    }
                }
            }
            (Tile::Cmma(_), RowwiseTileWorkspace::Bounce(bw)) => {
                bw.local_tile.rowwise_scale(scale);
            }
            _ => panic!("rowwise_scale: variant mismatch"),
        }
    }

    pub fn scale_and_mask<M: Mask>(
        &mut self,
        scale: E,
        mask: &M,
        ws: &mut RowwiseTileWorkspace<E>,
    ) {
        match (self, ws) {
            (Tile::Register(t), _) => {
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
            (Tile::Cmma(_), RowwiseTileWorkspace::Bounce(bw)) => {
                bw.local_tile.scale_and_mask::<M>(scale, mask);
            }
            _ => panic!("scale_and_mask: variant mismatch"),
        }
    }

    pub fn fill_zero(&mut self) {
        match self {
            Tile::Register(t) => {
                let m = comptime!(t.config.tile_size.m());
                let n = comptime!(t.config.tile_size.n());
                for i in 0..m * n {
                    t.data[i as usize] = E::from_int(0);
                }
            }
            Tile::Cmma(t) => {
                cmma::fill(&t.matrix, E::from_int(0));
            }
            _ => panic!("fill_zero: unsupported tile variant"),
        }
    }
}
