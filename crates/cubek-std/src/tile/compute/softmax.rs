use cubecl;
use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::compute::copy::{cmma_to_whitebox_fragment, whitebox_fragment_to_cmma};
use crate::tile::compute::mask::{Mask, MaskExpand};
use crate::tile::compute::rowwise::reducer::{fragment_row_max, fragment_row_sum};
use crate::tile::data::{
    BounceTile, InnerLayout, RegisterTile, RowWise, RowWiseExpand, UnitTile, WhiteboxFragment,
};
use crate::tile::{Plane, Tile, TileExpand};

/// Logits below this are considered masked (effectively -inf).
/// Value chosen to fit within f16 range (~-65,504 max).
pub const LOGIT_MASKED: f32 = -6e4;

/// Any value smaller than this is considered numerically zero (used for
/// fully-masked rows or tiny contributions). Value chosen to be above f16
/// smallest normal (~6.1e-5).
pub const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;

#[cube]
impl<E: Float> RowWise<E> {
    /// Replaces each value `v` (v >= 0) in a row with `1/v`.
    ///
    /// If `v = 0`, the result is set to `0` instead of `1/0`.
    /// This occurs when the entire row is masked, meaning it should
    /// contribute no information, and ensures numerical stability.
    pub fn recip_inplace(&mut self) {
        for i in 0..self.num_rows {
            let row_val = self.vals[i];

            let epsilon = E::new(FULLY_MASKED_ROW_THRESHOLD);
            let not_masked = E::cast_from(row_val >= epsilon);
            let safe_val = clamp_min(row_val, epsilon);
            let recip = safe_val.recip();
            self.vals[i] = not_masked * recip;
        }
    }
}

/// Comptime descriptor for the row-shape used by online softmax. Determines
/// how many rows per unit each running-state vector holds.
///
/// - `Direct { num_rows_per_unit }` — used with `Tile::Unit` or `Tile::Register`
///   when each unit owns its own copy of the tile.
/// - `Plane { inner_layout }` — used with `Tile::WhiteboxFragment` or `Tile::Bounce`,
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
    /// precision-cast write into a value-matmul lhs tile. Dispatches on the
    /// score variant — each variant owns the algorithm best suited to its
    /// storage and is polymorphic in the destination: a `Bounce` score can
    /// be written into any compatible softmaxed tile (Bounce, fragment, …),
    /// not just another `Bounce`.
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
        match self {
            Tile::Bounce(s) => {
                bounce_softmax::<Acc, Lhs, M>(s, softmaxed_tile, mask, state, head_dim_factor)
            }
            Tile::WhiteboxFragment(s) => {
                fragment_softmax::<Acc, Lhs, M>(s, softmaxed_tile, mask, state, head_dim_factor)
            }
            Tile::Unit(s) => {
                unit_softmax::<Acc, Lhs, M>(s, softmaxed_tile, mask, state, head_dim_factor)
            }
            Tile::Register(s) => {
                register_softmax::<Acc, Lhs, M>(s, softmaxed_tile, mask, state, head_dim_factor)
            }
            _ => panic!("softmax: unsupported score variant"),
        }
    }

    /// Multiplies each row of `self` by the corresponding `scale[r]`. The
    /// `Bounce` arm round-trips through smem so the cmma fragment is current
    /// for the next mma; the others operate in place on their native storage.
    pub fn scale_mul<SM: Float>(&mut self, scale: &RowWise<SM>) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        match self {
            Tile::Bounce(b) => {
                cmma_to_whitebox_fragment::<Acc>(b);
                b.fragment.rowwise_scale(&scale_acc);
                whitebox_fragment_to_cmma::<Acc>(b);
            }
            Tile::WhiteboxFragment(t) => t.rowwise_scale(&scale_acc),
            Tile::Unit(t) => t.rowwise_scale(&scale_acc),
            Tile::Register(t) => register_rowwise_scale::<Acc>(t, &scale_acc),
            _ => panic!("scale_mul: unsupported tile variant"),
        }
    }

    /// Divides each row of `self` by the corresponding `running_state_l[r]`,
    /// guarding against zero (a fully-masked row stays zero).
    pub fn scale_div<SM: Float>(&mut self, running_state_l: &RowWise<SM>) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(running_state_l);
        scale.recip_inplace();
        match self {
            Tile::Bounce(b) => {
                cmma_to_whitebox_fragment::<Acc>(b);
                b.fragment.rowwise_scale(&scale);
                whitebox_fragment_to_cmma::<Acc>(b);
            }
            Tile::WhiteboxFragment(t) => t.rowwise_scale(&scale),
            Tile::Unit(t) => t.rowwise_scale(&scale),
            Tile::Register(t) => register_rowwise_scale::<Acc>(t, &scale),
            _ => panic!("scale_div: unsupported tile variant"),
        }
    }

    /// Copies `self` into `dest` (a stage-side strided/shared tile in the
    /// caller's downstream write path).
    pub fn write_results<DE: Float, DS: Size>(&self, dest: &mut Tile<DE, Plane, ReadWrite>) {
        dest.copy_from::<Acc, DS, Acc, Acc, Acc, ReadWrite>(self, StageIdent::Out);
    }
}

#[cube]
fn bounce_softmax<Acc: Float, Lhs: Float, M: Mask>(
    score: &mut BounceTile<Acc>,
    softmaxed: &mut Tile<Lhs, Plane, ReadWrite>,
    mask: &M,
    state: &mut (RowWise<Acc>, RowWise<Acc>),
    head_dim_factor: Acc,
) -> RowWise<Acc> {
    let num_rows = comptime!(state.0.num_rows);
    let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
    let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

    // cmma → fragment once at entry so all subsequent ops read/write the
    // fragment view.
    cmma_to_whitebox_fragment::<Acc>(score);

    score.fragment.scale_and_mask::<M>(head_dim_factor, mask);
    fragment_row_max::<Acc>(&mut max_buf, &state.0, &score.fragment);
    score.fragment.exp_diff(&max_buf);
    fragment_row_sum::<Acc>(&mut sum_buf, &score.fragment);

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    // The post-exp values are still in `score.fragment` — we skip
    // `whitebox_fragment_to_cmma` on score (its cmma is cleared next
    // iteration) and stream the values straight into `softmaxed`.
    write_fragment_into::<Acc, Lhs>(&score.fragment, softmaxed);

    RowWise::copy_from(&mut state.0, &max_buf);
    RowWise::copy_from(&mut state.1, &new_l);

    exp_m_diff
}

#[cube]
fn fragment_softmax<Acc: Float, Lhs: Float, M: Mask>(
    score: &mut WhiteboxFragment<Acc>,
    softmaxed: &mut Tile<Lhs, Plane, ReadWrite>,
    mask: &M,
    state: &mut (RowWise<Acc>, RowWise<Acc>),
    head_dim_factor: Acc,
) -> RowWise<Acc> {
    let num_rows = comptime!(state.0.num_rows);
    let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
    let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

    score.scale_and_mask::<M>(head_dim_factor, mask);
    fragment_row_max::<Acc>(&mut max_buf, &state.0, score);
    score.exp_diff(&max_buf);
    fragment_row_sum::<Acc>(&mut sum_buf, score);

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    write_fragment_into::<Acc, Lhs>(score, softmaxed);

    RowWise::copy_from(&mut state.0, &max_buf);
    RowWise::copy_from(&mut state.1, &new_l);

    exp_m_diff
}

#[cube]
fn unit_softmax<Acc: Float, Lhs: Float, M: Mask>(
    score: &mut UnitTile<Acc>,
    softmaxed: &mut Tile<Lhs, Plane, ReadWrite>,
    mask: &M,
    state: &mut (RowWise<Acc>, RowWise<Acc>),
    head_dim_factor: Acc,
) -> RowWise<Acc> {
    let num_rows = comptime!(state.0.num_rows);
    let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
    let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

    score.scale_and_mask::<M>(head_dim_factor, mask);

    max_buf.copy_from(&state.0);
    max_buf.max_inplace(&score.rowwise_max());

    score.exp_diff(&max_buf);

    sum_buf.add_inplace(&score.rowwise_sum());

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    match softmaxed {
        Tile::Unit(d) => write_unit_into::<Acc, Lhs>(score, d),
        Tile::Bounce(_) => panic!("unit_softmax: Bounce destination not supported"),
        Tile::WhiteboxFragment(_) => {
            panic!("unit_softmax: WhiteboxFragment destination not supported")
        }
        Tile::Register(_) => panic!("unit_softmax: Register destination not supported"),
        _ => panic!("unit_softmax: unsupported softmaxed variant"),
    }

    RowWise::copy_from(&mut state.0, &max_buf);
    RowWise::copy_from(&mut state.1, &new_l);

    exp_m_diff
}

#[cube]
fn register_softmax<Acc: Float, Lhs: Float, M: Mask>(
    score: &mut RegisterTile<Acc>,
    softmaxed: &mut Tile<Lhs, Plane, ReadWrite>,
    mask: &M,
    state: &mut (RowWise<Acc>, RowWise<Acc>),
    head_dim_factor: Acc,
) -> RowWise<Acc> {
    let m = comptime!(score.config.tile_size.m());
    let n = comptime!(score.config.tile_size.n());
    let num_rows = comptime!(state.0.num_rows);
    let threshold = Acc::new(LOGIT_MASKED);

    let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
    let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

    for r in 0..m {
        let row_offset = r * n;
        for c in 0..n {
            let idx = (row_offset + c) as usize;
            score.data[idx] = score.data[idx] * head_dim_factor
                + Acc::cast_from(mask.should_mask((r, c))) * Acc::min_value();
        }
    }

    max_buf.copy_from(&state.0);
    for r in 0..m as usize {
        let row_offset = r as u32 * n;
        let mut val = Acc::min_value();
        for c in 0..n {
            val = max(val, score.data[(row_offset + c) as usize]);
        }
        max_buf.vals[r] = max(max_buf.vals[r], val);
    }

    for r in 0..m as usize {
        let row_offset = r as u32 * n;
        let val = max_buf.vals[r];
        let safe_val = clamp_min(val, threshold);
        let not_masked = Acc::cast_from(val >= threshold);
        for c in 0..n {
            let idx = (row_offset + c) as usize;
            score.data[idx] = not_masked * (score.data[idx] - safe_val).exp();
        }
    }

    for r in 0..m as usize {
        let row_offset = r as u32 * n;
        let mut val = Acc::from_int(0);
        for c in 0..n {
            val += score.data[(row_offset + c) as usize];
        }
        sum_buf.vals[r] += val;
    }

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    match softmaxed {
        Tile::Register(d) => write_register_into::<Acc, Lhs>(score, d),
        Tile::Bounce(_) => panic!("register_softmax: Bounce destination not supported"),
        Tile::WhiteboxFragment(_) => {
            panic!("register_softmax: WhiteboxFragment destination not supported")
        }
        Tile::Unit(_) => panic!("register_softmax: Unit destination not supported"),
        _ => panic!("register_softmax: unsupported softmaxed variant"),
    }

    RowWise::copy_from(&mut state.0, &max_buf);
    RowWise::copy_from(&mut state.1, &new_l);

    exp_m_diff
}

/// Writes a `WhiteboxFragment` of post-softmax values into `softmaxed`,
/// dispatching on the destination variant. The source is plane-fragmented so
/// each unit only writes its slice; for a `Bounce` destination this routes
/// directly through its smem into its cmma fragment.
#[cube]
fn write_fragment_into<Acc: Float, Lhs: Float>(
    src: &WhiteboxFragment<Acc>,
    softmaxed: &mut Tile<Lhs, Plane, ReadWrite>,
) {
    match softmaxed {
        Tile::Bounce(d) => {
            let stride = comptime!(d.cmma.tile_size.n());
            src.store_to(&mut d.smem);
            sync_cube();
            cubecl::cmma::load(&d.cmma.matrix, &d.smem.to_slice(), stride);
        }
        Tile::WhiteboxFragment(d) => {
            let total = comptime!(src.layout.unit_size.0 * src.layout.unit_size.1);
            for i in 0..total {
                d.array[i as usize] = Lhs::cast_from(src.array[i as usize]);
            }
        }
        _ => panic!("write_fragment_into: unsupported softmaxed variant"),
    }
}

#[cube]
fn write_unit_into<Acc: Float, Lhs: Float>(src: &UnitTile<Acc>, dest: &mut UnitTile<Lhs>) {
    let total = comptime!(src.layout.num_rows * src.layout.num_cols);
    for i in 0..total {
        dest.data[i as usize] = Lhs::cast_from(src.data[i as usize]);
    }
}

#[cube]
fn write_register_into<Acc: Float, Lhs: Float>(
    src: &RegisterTile<Acc>,
    dest: &mut RegisterTile<Lhs>,
) {
    let m = comptime!(src.config.tile_size.m());
    let n = comptime!(src.config.tile_size.n());
    for i in 0..m * n {
        dest.data[i as usize] = Lhs::cast_from(src.data[i as usize]);
    }
}

#[cube]
fn register_rowwise_scale<E: Float>(tile: &mut RegisterTile<E>, scale: &RowWise<E>) {
    let m = comptime!(tile.config.tile_size.m());
    let n = comptime!(tile.config.tile_size.n());
    for r in 0..m as usize {
        let row_offset = r as u32 * n;
        for c in 0..n {
            let idx = (row_offset + c) as usize;
            tile.data[idx] = tile.data[idx] * scale.vals[r];
        }
    }
}
