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
    /// ([`Scan`](crate::load::transport::base::Scan)); every claim this rests on was asserted there.
    pub(crate) fn fill_scanned<W: Size>(&mut self, src: &Memory<T>, #[comptime] scan: Scan) {
        match comptime!(scan) {
            // The source's served element, one line for one line.
            Scan::Element => self.scan_transparent::<T, W, W>(src),
            // `u32` words, one for one: the values unpack as they are read.
            Scan::Words => {
                let size!(WP) = comptime!(src.store.packing.physical(src.store.vector_size));
                self.scan_transparent::<u32, WP, W>(src)
            }
            // `i8` codes under a scheme, served and stored being the same line.
            Scan::Codes => self.scan_transparent::<i8, W, W>(src),
            // The source's line is one whole word and this stage is narrower: unpack each word
            // across several lines.
            Scan::SubWord => self.scan_words::<W>(src),
        }
    }

    /// The cooperative flat scan behind [`fill_from`](Memory::fill_from)'s general path: cyclic
    /// across the cube, each unit writing lines `u`, `u + CUBE_DIM`, …, read at storage element
    /// `I` via [`flat_transparent`](Memory::flat_transparent), so a quantized source dequantizes.
    pub(crate) fn scan_transparent<I: Numeric, WP: Size, W: Size>(&mut self, src: &Memory<T>) {
        let s = src.flat_transparent::<I, WP, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        // One line per unit, striding by the cube: **the deal is disjoint**, which a folding
        // destination rests on. A repeated deal would be invisible under `Write::Replace`
        // and land once per repeat under `Write::Accumulate`; any future deal owes the same.
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
