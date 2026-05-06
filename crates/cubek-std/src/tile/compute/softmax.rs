use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::compute::mask::Mask;
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
                b.cmma_to_fragment();
                b.rowwise_scale(&scale_acc);
                b.fragment_to_cmma();
            }
            Tile::WhiteboxFragment(t) => t.rowwise_scale(&scale_acc),
            Tile::Unit(t) => t.rowwise_scale(&scale_acc),
            Tile::Register(t) => t.rowwise_scale(&scale_acc),
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
                b.cmma_to_fragment();
                b.rowwise_scale(&scale);
                b.fragment_to_cmma();
            }
            Tile::WhiteboxFragment(t) => t.rowwise_scale(&scale),
            Tile::Unit(t) => t.rowwise_scale(&scale),
            Tile::Register(t) => t.rowwise_scale(&scale),
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
    // fragment view. The post-exp values are still in the fragment at the end —
    // we skip `fragment_to_cmma` on `score` (its cmma is cleared next iteration)
    // and stream straight into `softmaxed`.
    score.cmma_to_fragment();

    score.scale_and_mask::<M>(head_dim_factor, mask);
    score.row_max(&mut max_buf, &state.0);
    score.exp_diff(&max_buf);
    score.row_sum(&mut sum_buf);

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    score.write_fragment_to::<Lhs, Plane>(softmaxed);

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
    score.row_max(&mut max_buf, &state.0);
    score.exp_diff(&max_buf);
    score.row_sum(&mut sum_buf);

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
    score.row_max(&mut max_buf, &state.0);
    score.exp_diff(&max_buf);
    score.row_sum(&mut sum_buf);

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    match softmaxed {
        Tile::Unit(d) => score.write_to::<Lhs>(d),
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
    let num_rows = comptime!(state.0.num_rows);
    let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
    let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

    score.scale_and_mask::<M>(head_dim_factor, mask);
    score.row_max(&mut max_buf, &state.0);
    score.exp_diff(&max_buf);
    score.row_sum(&mut sum_buf);

    let exp_m_diff = state.0.exp_diff(&max_buf);
    let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

    match softmaxed {
        Tile::Register(d) => score.write_to::<Lhs>(d),
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

/// Writes a free-standing `WhiteboxFragment` of post-softmax values into
/// `softmaxed`. Used by `fragment_softmax`; the `Bounce` source path lives on
/// `BounceTile::write_fragment_to`.
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
