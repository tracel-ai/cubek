//! The row ops at plane ownership: a plane owns a row-slice, its lanes split
//! the reduced axis, and each row's reduction closes in one plane instruction.
//! The [`rowwise`](super::rowwise) twin runs the same algebra with a unit
//! owning the whole row; which one a call reaches is the state's
//! [`RowShare`] and nothing else.
//!
//! A lane touches only the columns `lane, lane + lanes, …` of its plane's rows,
//! in every op — so nothing here reads a cell another lane wrote, and the leaf
//! keeps the twin's promise of no syncs. What crosses lanes is the reduced
//! scalar, and it crosses through the hardware.
//!
//! **The plane must be the cube's**: `lanes` is the width the device commits
//! to, and a plane may not straddle the x dim's teams, so `CUBE_DIM_X` has to
//! be a whole number of planes. A wrong width reduces over the wrong lanes and
//! is silently wrong, which is why the caller states it rather than reads it.

use cubecl::prelude::*;

use crate::{instruction::Monoid, instruction::plane, *};

#[cube]
impl<EA: Float> Tile<EA> {
    /// [`scale_and_mask`](Tile::scale_and_mask) at plane ownership.
    pub fn scale_and_mask_planar(
        &mut self,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        #[comptime] rpp: usize,
        #[comptime] lanes: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<W>();

        for ri in 0..rpp {
            let r = plane_row(ri, rpp, lanes);
            if r < rows {
                let q = probe.row_q(r);
                let mut c = lane(lanes);
                while c < cols {
                    let masked = probe.masked(q, probe.origin_s + c, mask);
                    let val = select(
                        masked,
                        EA::min_value(),
                        view.read(r * cols + c).extract(0usize) * scale,
                    );
                    view.write(r * cols + c, Vector::cast_from(val));
                    c += lanes;
                }
            }
        }
    }

    /// [`row_max`](Tile::row_max) at plane ownership: a lane's partial over its
    /// own columns, then one plane reduction per row. Seeding with `base` on
    /// every lane is free — a max is idempotent, so the seed survives the fold
    /// whichever lane carried it.
    pub fn row_max_planar(
        &self,
        acc: &mut Array<EA>,
        base: &Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] lanes: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let view = self.flat::<W>();

        for ri in 0..rpp {
            let mut partial = base[ri];
            let r = plane_row(ri, rpp, lanes);
            if r < rows {
                let mut c = lane(lanes);
                while c < cols {
                    partial = max(partial, view.read(r * cols + c).extract(0usize));
                    c += lanes;
                }
            }
            acc[ri] = plane::reduce::<EA>(partial, lanes, comptime!(Monoid::Max));
        }
    }

    /// [`exp_diff`](Tile::exp_diff) at plane ownership. `rowwise` is
    /// plane-uniform coming out of [`row_max_planar`](Tile::row_max_planar),
    /// so every lane exponentiates against the same row max.
    pub fn exp_diff_planar(
        &mut self,
        rowwise: &Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] lanes: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<W>();

        for ri in 0..rpp {
            let r = plane_row(ri, rpp, lanes);
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                let mut c = lane(lanes);
                while c < cols {
                    let e = live * (view.read(r * cols + c).extract(0usize) - safe_m).exp();
                    view.write(r * cols + c, Vector::cast_from(e));
                    c += lanes;
                }
            }
        }
    }

    /// [`row_sum`](Tile::row_sum) at plane ownership. Unlike the max there is
    /// no seed: a sum's identity is zero and every lane must contribute its
    /// own columns exactly once.
    pub fn row_sum_planar(
        &self,
        acc: &mut Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] lanes: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let view = self.flat::<W>();

        for ri in 0..rpp {
            let mut partial = EA::from_int(0);
            let r = plane_row(ri, rpp, lanes);
            if r < rows {
                let mut c = lane(lanes);
                while c < cols {
                    partial += view.read(r * cols + c).extract(0usize);
                    c += lanes;
                }
            }
            acc[ri] = plane::reduce::<EA>(partial, lanes, comptime!(Monoid::Sum));
        }
    }

    /// [`write_rows_to`](Tile::write_rows_to) at plane ownership.
    pub(crate) fn write_rows_to_planar<EP: Numeric>(
        &self,
        dest: &mut Tile<EP>,
        #[comptime] rpp: usize,
        #[comptime] lanes: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let size!(WP) = dest.vector_size();
        let src = self.flat::<W>();
        let mut dst = dest.flat_mut::<WP>();

        for ri in 0..rpp {
            let r = plane_row(ri, rpp, lanes);
            if r < rows {
                let mut c = lane(lanes);
                while c < cols {
                    dst.write(
                        r * cols + c,
                        Vector::cast_from(src.read(r * cols + c).extract(0usize)),
                    );
                    c += lanes;
                }
            }
        }
    }
}

/// This lane's index within its plane.
#[cube]
fn lane(#[comptime] lanes: usize) -> usize {
    UNIT_POS_X as usize % lanes
}

/// The score row this plane's `ri`-th slot owns. Plane-uniform by
/// construction, so a guard on it never splits a plane and the reductions
/// below it are reached by every lane.
#[cube]
fn plane_row(ri: usize, #[comptime] rpp: usize, #[comptime] lanes: usize) -> usize {
    (UNIT_POS_X as usize / lanes) * rpp + ri
}
