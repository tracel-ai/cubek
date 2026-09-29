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

    /// Cooperative cyclic flat scan across the cube; a packed source unpacks.
    pub(crate) fn scan_unpacked<WP: Size, W: Size>(&mut self, src: &Memory<T>) {
        let s = src.flat_unpacked::<WP, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        // One line per unit, striding by the cube: the distribution must stay disjoint, or a
        // `Write::Accumulate` destination folds a value more than once.
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            // `src` zeroes reads past its bound; the staged buffer writes the full padded cell.
            d.write(i, s.read(i));
            i += workers;
        }
    }
}
