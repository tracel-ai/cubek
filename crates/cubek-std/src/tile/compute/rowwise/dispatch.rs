use cubecl::prelude::*;

use crate::tile::{
    LOGIT_MASKED, Plane, Tile, TileExpand,
    compute::rowwise::reducer::{fragment_row_max, fragment_row_sum},
    data::RowWise,
};

/// Row-wise primitives on a `Tile<E, Plane, ReadWrite>` used for attention's
/// online softmax and output scaling. Dispatch happens per-variant:
/// - `Tile::Unit` — each unit holds its own copy of the tile, ops run in
///   registers.
/// - `Tile::WhiteboxFragment` — the tile is fragmented across plane units with
///   an exposed layout, row-reductions use `plane_shuffle`.
/// - `Tile::Bounce` — same as `WhiteboxFragment` but the underlying compute
///   fragment (cmma) is opaque. The row-wise ops here read/write the inner
///   fragment view; the smem ↔ cmma synchronization is driven by the higher-
///   level `softmax` / `scale_mul` / `scale_div` methods (see `softmax.rs`).
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
            Tile::WhiteboxFragment(t) => {
                fragment_row_max::<E>(acc, base, t);
            }
            Tile::Bounce(b) => {
                fragment_row_max::<E>(acc, base, &b.fragment);
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
            Tile::WhiteboxFragment(t) => {
                fragment_row_sum::<E>(acc, t);
            }
            Tile::Bounce(b) => {
                fragment_row_sum::<E>(acc, &b.fragment);
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
            Tile::WhiteboxFragment(t) => t.exp_diff(rowwise),
            Tile::Bounce(b) => b.fragment.exp_diff(rowwise),
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
            Tile::WhiteboxFragment(t) => t.rowwise_scale(scale),
            Tile::Bounce(b) => b.fragment.rowwise_scale(scale),
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
}
