use cubecl;
use cubecl::prelude::*;

use crate::tile::ops::broadcast_reducer::{local_row_max, local_row_sum};
use crate::tile::ops::{LOGIT_MASKED, Mask, MaskExpand, RowWise};
use crate::tile::variants::BounceTile;
use crate::tile::{Plane, Tile, TileExpand};

/// Row-wise primitives on a `Tile<E, Plane, ReadWrite>` used for attention's
/// online softmax and output scaling. Dispatch happens per-variant:
/// - `Tile::Unit` — each unit holds its own copy of the tile, ops run in
///   registers.
/// - `Tile::Local` — the tile is fragmented across plane units, row-reductions
///   use `plane_shuffle`.
/// - `Tile::Bounce` — same as `Local` but the underlying compute fragment
///   (cmma) is opaque. The row-wise ops here read/write the inner `LocalTile`;
///   the smem ↔ cmma synchronization is driven by the higher-level
///   `softmax` / `scale_mul` / `scale_div` methods (see `ops/softmax.rs`).
/// - `Tile::Register` — kept for the legacy direct-register attention path.
#[cube]
impl<E: Float> Tile<E, Plane, ReadWrite> {
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        match self {
            Tile::Unit(t) => {
                acc.copy_from(base);
                let m = comptime!(t.layout.num_rows);
                let n = comptime!(t.layout.num_cols);
                for r in 0..m as usize {
                    let row_offset = r as u32 * n;
                    let mut val = E::min_value();
                    for c in 0..n {
                        val = max(val, t.data[(row_offset + c) as usize]);
                    }
                    acc.vals[r] = max(acc.vals[r], val);
                }
            }
            Tile::Local(t) => {
                local_row_max::<E>(acc, base, t);
            }
            Tile::Bounce(b) => {
                local_row_max::<E>(acc, base, &b.local);
            }
            Tile::Register(t) => {
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
            _ => panic!("row_max: unsupported tile variant"),
        }
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        match self {
            Tile::Unit(t) => {
                acc.fill(E::from_int(0));
                let m = comptime!(t.layout.num_rows);
                let n = comptime!(t.layout.num_cols);
                for r in 0..m as usize {
                    let row_offset = r as u32 * n;
                    let mut val = E::from_int(0);
                    for c in 0..n {
                        val += t.data[(row_offset + c) as usize];
                    }
                    acc.vals[r] += val;
                }
            }
            Tile::Local(t) => {
                local_row_sum::<E>(acc, t);
            }
            Tile::Bounce(b) => {
                local_row_sum::<E>(acc, &b.local);
            }
            Tile::Register(t) => {
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
            _ => panic!("row_sum: unsupported tile variant"),
        }
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        match self {
            Tile::Unit(t) => t.exp_diff(rowwise),
            Tile::Local(t) => t.exp_diff(rowwise),
            Tile::Bounce(b) => b.local.exp_diff(rowwise),
            Tile::Register(t) => {
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
            _ => panic!("exp_diff: unsupported tile variant"),
        }
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        match self {
            Tile::Unit(t) => t.rowwise_scale(scale),
            Tile::Local(t) => t.rowwise_scale(scale),
            Tile::Bounce(b) => b.local.rowwise_scale(scale),
            Tile::Register(t) => {
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
            _ => panic!("rowwise_scale: unsupported tile variant"),
        }
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        match self {
            Tile::Unit(t) => t.scale_and_mask::<M>(scale, mask),
            Tile::Local(t) => t.scale_and_mask::<M>(scale, mask),
            Tile::Bounce(b) => b.local.scale_and_mask::<M>(scale, mask),
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
            Tile::Local(t) => t.zero(),
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

/// Internal `copy_from` between the `cmma` and `local` parts of a [`BounceTile`]:
/// cmma -> smem -> local. Used by the high-level `softmax` / `scale_mul` /
/// `scale_div` methods to make the local view current.
#[cube]
pub(crate) fn cmma_to_local<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    cubecl::cmma::store(
        &mut b.smem,
        &b.cmma.matrix,
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    b.local.load_from_slice(&b.smem.to_slice());
    sync_cube();
}

/// Internal `copy_from` between the `local` and `cmma` parts of a [`BounceTile`]:
/// local -> smem -> cmma. Reverses [`cmma_to_local`].
#[cube]
pub(crate) fn local_to_cmma<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    b.local.store_to(&mut b.smem);
    sync_cube();
    cubecl::cmma::load_with_layout(
        &b.cmma.matrix,
        &b.smem.to_slice(),
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
}
