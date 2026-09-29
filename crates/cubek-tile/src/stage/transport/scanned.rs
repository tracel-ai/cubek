//! The scanned transport: source and destination each read as a flat run of its own window,
//! which pairs them only where the two are the same box.

use cubecl::prelude::*;

use super::base::Scan;
use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// The general path of [`fill_from`](Memory::fill_from): source and destination each scanned
    /// as a flat run of its own *window*, which pairs them only when they are the same box.
    ///
    /// `scan` is what the source hands out per line, read off the two forms once
    /// ([`Scan`](crate::stage::transport::base::Scan)); every claim this rests on was asserted there.
    pub(crate) fn fill_scanned<W: Size>(&mut self, src: &Memory<T>, #[comptime] scan: Scan) {
        match comptime!(scan) {
            // The source's served element, one line for one line.
            Scan::Element => self.scan_unpacked::<W, W>(src),
            // `u32` words, one for one: the values unpack as they are read.
            Scan::Words => {
                let size!(WP) = comptime!(src.store.packing.physical(src.store.vector_size));
                self.scan_unpacked::<WP, W>(src)
            }
        }
    }

    /// The cooperative flat scan behind [`fill_from`](Memory::fill_from)'s general path: cyclic
    /// across the cube, each unit writing lines `u`, `u + CUBE_DIM`, …, read through
    /// [`flat_unpacked`](Memory::flat_unpacked), so a packed source unpacks.
    pub(crate) fn scan_unpacked<WP: Size, W: Size>(&mut self, src: &Memory<T>) {
        let s = src.flat_unpacked::<WP, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        // One line per unit, striding by the cube: **the distribution is disjoint**, which a folding
        // destination rests on. A repeated distribution would be invisible under `Write::Replace`
        // and land once per repeat under `Write::Accumulate`; any future distribution owes the same.
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            // `src` zeroes reads past its logical bound (the partial-tile overhang); the
            // staged buffer is unchecked, so the full padded cell is still written.
            d.write(i, s.read(i));
            i += workers;
        }
    }
}
