use cubecl;
use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::ops::tile_ops::{cmma_to_local, local_to_cmma};
use crate::tile::ops::{Mask, RowWise};
use crate::tile::variants::InnerLayout;
use crate::tile::{Plane, Tile, TileExpand};

/// Comptime descriptor for the row-shape used by online softmax. Determines
/// how many rows per unit each running-state vector holds.
///
/// - `Direct { num_rows_per_unit }` — used with `Tile::Unit` or `Tile::Register`
///   when each unit owns its own copy of the tile.
/// - `Plane { inner_layout }` — used with `Tile::Local` or `Tile::Bounce`,
///   where the inner layout determines how many rows each unit covers.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SoftmaxKind {
    Direct { num_rows_per_unit: u32 },
    Plane { inner_layout: InnerLayout },
}

impl SoftmaxKind {
    pub const fn num_rows_per_unit(&self) -> u32 {
        match self {
            SoftmaxKind::Direct { num_rows_per_unit } => *num_rows_per_unit,
            SoftmaxKind::Plane { inner_layout } => match inner_layout {
                InnerLayout::Contiguous => 1,
                InnerLayout::SplitRows => 2,
            },
        }
    }
}

/// Initial running state `(m, l)` for the online softmax over a single tile row.
#[cube]
pub fn softmax_init_state<E: Float>(
    #[comptime] num_rows_per_unit: u32,
) -> (RowWise<E>, RowWise<E>) {
    (
        RowWise::<E>::new_min_value(num_rows_per_unit as usize),
        RowWise::<E>::new_zero(num_rows_per_unit as usize),
    )
}

#[cube]
impl<Acc: Float> Tile<Acc, Plane, ReadWrite> {
    /// Online softmax update over a single attention tile, fused with the
    /// precision-cast write into a value-matmul lhs tile.
    ///
    /// For `Tile::Bounce`, the smem ↔ cmma sync is internal: `cmma_to_local`
    /// once at the start so all subsequent ops read/write the local view, and
    /// the softmaxed values are streamed straight into the destination's cmma
    /// fragment via its own smem (no `local_to_cmma` for `self`, which is
    /// cleared next iteration).
    ///
    /// Returns the per-row scaling factor `α_i = e^(m_old - m_new)` used by the
    /// caller to rescale running output accumulators.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed_tile: &mut Tile<Lhs, Plane, ReadWrite>,
        state: &mut (RowWise<Acc>, RowWise<Acc>),
        head_dim_factor: Acc,
    ) -> RowWise<Acc> {
        let num_rows = comptime!(state.0.num_rows);
        let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
        let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

        bounce_in(self);

        self.scale_and_mask::<M>(head_dim_factor, mask);
        self.row_max(&mut max_buf, &state.0);
        self.exp_diff(&max_buf);
        self.row_sum(&mut sum_buf);

        let exp_m_diff = state.0.exp_diff(&max_buf);
        let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

        write_softmaxed(self, softmaxed_tile);

        RowWise::copy_from(&mut state.0, &max_buf);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }

    /// Multiplies each row of `self` by the corresponding `scale[r]`. For
    /// `Tile::Bounce`, this round-trips through smem so the cmma fragment is
    /// up to date for the next mma.
    pub fn scale_mul<SM: Float>(&mut self, scale: &RowWise<SM>) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        bounce_in(self);
        self.rowwise_scale(&scale_acc);
        bounce_out(self);
    }

    /// Divides each row of `self` by the corresponding `running_state_l[r]`,
    /// guarding against zero (a fully-masked row stays zero).
    pub fn scale_div<SM: Float>(&mut self, running_state_l: &RowWise<SM>) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(running_state_l);
        scale.recip_inplace();
        bounce_in(self);
        self.rowwise_scale(&scale);
        bounce_out(self);
    }

    /// Copies `self` into `dest` (a stage-side strided/shared tile in the
    /// caller's downstream write path).
    pub fn write_results<DE: Float, DS: Size>(&self, dest: &mut Tile<DE, Plane, ReadWrite>) {
        dest.copy_from::<Acc, DS, Acc, Acc, Acc, ReadWrite>(self, StageIdent::Out);
    }
}

#[cube]
fn bounce_in<E: Float>(tile: &mut Tile<E, Plane, ReadWrite>) {
    match tile {
        Tile::Bounce(b) => {
            cmma_to_local::<E>(b);
        }
        Tile::Unit(_) => {}
        Tile::Local(_) => {}
        Tile::Register(_) => {}
        _ => panic!("bounce_in: unsupported tile variant"),
    }
}

#[cube]
fn bounce_out<E: Float>(tile: &mut Tile<E, Plane, ReadWrite>) {
    match tile {
        Tile::Bounce(b) => {
            local_to_cmma::<E>(b);
        }
        Tile::Unit(_) => {}
        Tile::Local(_) => {}
        Tile::Register(_) => {}
        _ => panic!("bounce_out: unsupported tile variant"),
    }
}

#[cube]
fn write_softmaxed<Acc: Float, Lhs: Float>(
    score_tile: &Tile<Acc, Plane, ReadWrite>,
    softmaxed_tile: &mut Tile<Lhs, Plane, ReadWrite>,
) {
    match (score_tile, softmaxed_tile) {
        (Tile::Register(s), Tile::Register(d)) => {
            let m = comptime!(s.config.tile_size.m());
            let n = comptime!(s.config.tile_size.n());
            for i in 0..m * n {
                d.data[i as usize] = Lhs::cast_from(s.data[i as usize]);
            }
        }
        (Tile::Unit(s), Tile::Unit(d)) => {
            let m = comptime!(s.layout.num_rows);
            let n = comptime!(s.layout.num_cols);
            for i in 0..m * n {
                d.data[i as usize] = Lhs::cast_from(s.data[i as usize]);
            }
        }
        (Tile::Bounce(s), Tile::Bounce(d)) => {
            // score's LocalTile already holds the post-exp_diff values; route
            // through `softmaxed`'s smem to avoid clobbering score's smem and
            // load directly into softmaxed's cmma fragment.
            let stride = comptime!(d.cmma.tile_size.n());
            s.local.store_to(&mut d.smem);
            sync_cube();
            cubecl::cmma::load(&d.cmma.matrix, &d.smem.to_slice(), stride);
        }
        _ => panic!("write_softmaxed: incompatible tile pair"),
    }
}
