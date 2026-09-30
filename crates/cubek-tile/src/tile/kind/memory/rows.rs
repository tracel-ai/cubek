//! A plane's window of shared memory, read row by row: the verbs [`Rows`] runs on it, the plane's
//! units taking each row's cells in turns and meeting on the row's partials.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<E: Float> Memory<E> {
    /// `self[r, c] = self[r, c] · scale + bias` at every cell of this window over `space`.
    pub(crate) fn scale_add_rows(
        &mut self,
        scale: E,
        bias: &Procedural<E>,
        #[comptime] space: Space,
    ) {
        let columns = comptime!(space.columns());
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<W>();
        #[unroll]
        for r in 0..comptime!(space.rows()) {
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                let at = r * columns + c;
                let cell = bias.at_cell(r as u32, c as u32, space.clone());
                view.write(
                    at,
                    Vector::cast_from(view.read(at).extract(0usize) * scale + cell),
                );
                c += PLANE_DIM as usize;
            }
        }
    }

    /// Each row's max over this window over `space`, starting from `seed`'s.
    pub(crate) fn row_maxima(&self, seed: &Array<E>, #[comptime] space: Space) -> Array<E> {
        let columns = comptime!(space.columns());
        let size!(W) = 1usize;
        let view = self.flat::<W>();
        let mut maxima = Array::<E>::new(comptime!(space.rows()));
        #[unroll]
        for r in 0..comptime!(space.rows()) {
            let mut partial = seed[r];
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                partial = max(partial, view.read(r * columns + c).extract(0usize));
                c += PLANE_DIM as usize;
            }
            maxima[r] = plane_max(partial);
        }
        maxima
    }

    /// `self[r, c] = exp(self[r, c] − rows[r])` over this window over `space`
    /// ([`Rows::exp_minus_cell`]).
    pub(crate) fn exp_minus_rows(&mut self, rows: &Array<E>, #[comptime] space: Space) {
        let columns = comptime!(space.columns());
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<W>();
        #[unroll]
        for r in 0..comptime!(space.rows()) {
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                let at = r * columns + c;
                let cell = Rows::<E>::exp_minus_cell(view.read(at).extract(0usize), rows[r]);
                view.write(at, Vector::cast_from(cell));
                c += PLANE_DIM as usize;
            }
        }
    }

    /// Each row's sum over this window over `space`.
    pub(crate) fn row_sums(&self, #[comptime] space: Space) -> Array<E> {
        let columns = comptime!(space.columns());
        let size!(W) = 1usize;
        let view = self.flat::<W>();
        let mut sums = Array::<E>::new(comptime!(space.rows()));
        #[unroll]
        for r in 0..comptime!(space.rows()) {
            let mut partial = E::from_int(0);
            let mut c = UNIT_POS_PLANE as usize;
            while c < columns {
                partial += view.read(r * columns + c).extract(0usize);
                c += PLANE_DIM as usize;
            }
            sums[r] = plane_sum(partial);
        }
        sums
    }

    /// `self[r, c] *= factors[r]` over this window over `space`.
    pub(crate) fn mul_rows(&mut self, factors: &Array<E>, #[comptime] space: Space) {
        let (columns, cells) = comptime!((space.columns(), space.cells()));
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<W>();
        let mut at = UNIT_POS_PLANE as usize;
        while at < cells {
            view.write(at, view.read(at) * Vector::cast_from(factors[at / columns]));
            at += PLANE_DIM as usize;
        }
    }
}
