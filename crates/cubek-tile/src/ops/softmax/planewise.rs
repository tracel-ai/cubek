//! The row ops at plane ownership: a plane owns every row of the tile it is handed (windowed per
//! plane by the kernel), its units split the reduced axis, and each row's reduction closes in one
//! plane op. In the [`rowwise`](super::rowwise) twin a unit owns the row; [`RowShare`] picks.
//!
//! A unit touches only the lines `unit, unit + units, …` of its plane's rows (a line being the
//! tile's vector width of adjacent columns), so nothing reads a cell another unit wrote and the
//! leaf keeps the twin's promise of no syncs; only the reduced scalar crosses units, in hardware.
//!
//! Every loop is over a comptime bound and unrolls; the edge compare compiles out when the units
//! divide the lines, which a fold sized to its plane arranges.
//!
//! **The plane must be the cube's**: `units` is the width the device commits to, and a plane may
//! not straddle the x dim, so `CUBE_DIM_X` has to be a whole number of planes. A wrong width
//! reduces over the wrong units, silently, which is why the caller states it rather than reads it.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// [`scale_and_mask`](Tile::scale_and_mask) at plane ownership.
    pub fn scale_and_mask_planar(
        &mut self,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpp {
            let r = ri;
            if r < rows {
                let q = probe.row_q(r);
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let i = r * lines + line;
                        let mut v = view.read(i) * Vector::<EA, W>::cast_from(scale);
                        #[unroll]
                        for j in 0..w {
                            let masked = probe.masked(q, probe.origin_s + line * w + j, mask);
                            v.insert(j, select(masked, EA::min_value(), v.extract(j)));
                        }
                        view.write(i, v);
                    }
                }
            }
        }
    }

    /// [`row_max`](Tile::row_max) at plane ownership: a unit's partial over its own lines, then
    /// one plane reduction per row. Seeding with `base` on every unit is free: a max is
    /// idempotent, so the seed survives the fold whichever unit carried it.
    pub fn row_max_planar(
        &self,
        acc: &mut Array<EA>,
        base: &Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<W>();

        #[unroll]
        for ri in 0..rpp {
            let mut partial = base[ri];
            let r = ri;
            if r < rows {
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let v = view.read(r * lines + line);
                        #[unroll]
                        for j in 0..w {
                            partial = max(partial, v.extract(j));
                        }
                    }
                }
            }
            acc[ri] = comptime!(UnitShare::of_units(units, units)).fold::<EA>(partial, Monoid::Max);
        }
    }

    /// [`exp_diff`](Tile::exp_diff) at plane ownership. `rowwise` is
    /// plane-uniform coming out of [`row_max_planar`](Tile::row_max_planar),
    /// so every unit exponentiates against the same row max.
    pub fn exp_diff_planar(
        &mut self,
        rowwise: &Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpp {
            let r = ri;
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let i = r * lines + line;
                        let mut v = view.read(i);
                        #[unroll]
                        for j in 0..w {
                            v.insert(j, live * (v.extract(j) - safe_m).exp());
                        }
                        view.write(i, v);
                    }
                }
            }
        }
    }

    /// [`row_sum`](Tile::row_sum) at plane ownership. Unlike the max there is
    /// no seed: a sum's identity is zero and every unit must contribute its
    /// own lines exactly once.
    pub fn row_sum_planar(
        &self,
        acc: &mut Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<W>();

        #[unroll]
        for ri in 0..rpp {
            let mut partial = EA::from_int(0);
            let r = ri;
            if r < rows {
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let v = view.read(r * lines + line);
                        #[unroll]
                        for j in 0..w {
                            partial += v.extract(j);
                        }
                    }
                }
            }
            acc[ri] = comptime!(UnitShare::of_units(units, units)).fold::<EA>(partial, Monoid::Sum);
        }
    }

    /// [`write_rows_to`](Tile::write_rows_to) at plane ownership. The
    /// destination is laid out in the same lines.
    pub(crate) fn write_rows_to_planar<EP: Numeric>(
        &self,
        dest: &mut Tile<EP>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let wp = dest.vector_size();
        comptime!(assert!(
            w == wp,
            "write_rows_to: the probabilities are laid out in the score's lines"
        ));
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let src = self.flat::<W>();
        let mut dst = dest.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpp {
            let r = ri;
            if r < rows {
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let i = r * lines + line;
                        dst.write(i, Vector::<EP, W>::cast_from(src.read(i)));
                    }
                }
            }
        }
    }
}

/// This unit's index within its plane.
#[cube]
fn unit(#[comptime] units: usize) -> usize {
    UNIT_POS_X as usize % units
}
