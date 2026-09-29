//! The scanned transport: source and destination each read as a flat run of its own window.

use cubecl::prelude::*;

use super::base::Scan;
use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// General path of [`fill_from`](Memory::fill_from); `scan` is what the source hands out.
    pub(crate) fn fill_scanned<W: Size>(&mut self, src: &Memory<T>, #[comptime] scan: Scan) {
        match comptime!(scan) {
            Scan::Element => self.scan_unpacked::<W, W>(src),
            Scan::Words => {
                let size!(WP) = comptime!(src.store.packing.physical(src.store.vector_size));
                self.scan_unpacked::<WP, W>(src)
            }
        }
    }

    /// Cooperative cyclic flat scan across the units that share this fill ([`FillUnits`]); a
    /// packed source unpacks.
    pub(crate) fn scan_unpacked<WP: Size, W: Size>(&mut self, src: &Memory<T>) {
        let fill = comptime!(self.access.fill);
        let s = src.flat_unpacked::<WP, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        // One line per unit, striding by the units that fill: the distribution must stay
        // disjoint, or a `Write::Accumulate` destination folds a value more than once.
        let stride = fill_workers(fill);
        let mut i = fill_worker(fill);
        while i < total {
            // `src` zeroes reads past its bound; the staged buffer writes the full padded cell.
            d.write(i, s.read(i));
            i += stride;
        }
    }

    /// A copy of `src` into this window with a cast, `S` down (or up) to `T`, one line for one
    /// line: what a sum held in a wider buffer becomes when it lands in its output. The two are
    /// the same box, plain, and served at one width; the same cyclic scan as
    /// [`fill_scanned`](Memory::fill_scanned), each unit casting the lines it moves.
    pub(crate) fn fill_cast_from<S: Numeric>(&mut self, src: &Memory<S>) {
        comptime!(assert!(
            src.store.packing == Packing::Plain
                && self.store.packing == Packing::Plain
                && src.store.vector_size == self.store.vector_size
                && src.projection.is_direct()
                && self.projection.is_direct(),
            "Memory::fill_cast_from: a cast copy moves plain lines between two boxes served at \
             one width"
        ));
        let size!(W) = comptime!(self.store.vector_size);
        let s = src.flat_unpacked::<W, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            d.write(i, Vector::<T, W>::cast_from(s.read(i)));
            i += workers;
        }
    }
}
