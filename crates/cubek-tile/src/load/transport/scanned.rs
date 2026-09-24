//! The scanned transport: source and destination each read as a flat run of its own window,
//! which pairs them only where the two are the same box.

use cubecl::prelude::*;

use super::cooperative::fill_extent;
use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// The general path of [`fill_from`](Memory::fill_from): source and destination each scanned
    /// as a flat run of its own *window*, which pairs them only when they are the same box; a
    /// gathered side is refused. Taken by a windowed or masked destination, or a quantized source.
    pub(crate) fn fill_scanned<W: Size>(&mut self, src: &Memory<T>, #[comptime] space: Space) {
        let gathered = comptime!(!src.projection.is_direct());
        comptime!(assert!(
            !gathered && self.projection.is_direct(),
            "Memory::fill_from: a gathered tile fills only a whole, unmasked, unquantized \
             destination (a stage)"
        ));
        // The read decodes at the source's true storage element: `T` for a plain tile, else the
        // quantized store's element recovered from its scheme (`I` was erased at construction).
        // That lets a plain `copy_from`/`fill` dequantize on its own; the kernel never threads `I`.
        {
            #[comptime]
            match &src.store.quant {
                ComptimeOption::None => {
                    comptime!(assert!(
                        self.store.vector_size == src.store.vector_size,
                        "Memory::fill_from: a plain source is scanned at the destination's width, \
                         so a padded stage has to take the straight fill"
                    ));
                    // Equal widths here, so this only asks that the innermost extent is whole
                    // lines: `storage_extents` rounds it up, and nothing on the scan path would
                    // otherwise notice the last line the source cannot fill.
                    comptime!(fill_extent(
                        &space,
                        src.store.vector_size,
                        self.store.vector_size,
                        src.access.overhang.masks()
                    ));
                    // No scales to fold in, so the source's own packing is the whole read: served
                    // as stored, or unpacked from the words it holds into this plain destination.
                    let packing = src.packing();
                    match comptime!(packing) {
                        Packing::Plain => self.scan_transparent::<T, W, W>(src),
                        Packing::Native => panic!(
                            "Memory::fill_from: a native store with nothing to fold in serves \
                             its own element; bind it as that element"
                        ),
                        Packing::Packed { field: _ } => {
                            let size!(WP) = comptime!(packing.physical(src.store.vector_size));
                            self.scan_transparent::<u32, WP, W>(src)
                        }
                    }
                }
                ComptimeOption::Some(_) => match comptime!(src.store.packing) {
                    Packing::Native => self.scan_transparent::<i8, W, W>(src),
                    Packing::Packed { field: _ } => {
                        if comptime!(src.store.vector_size == self.store.vector_size) {
                            let size!(WP) =
                                comptime!(src.store.packing.physical(src.store.vector_size));
                            self.scan_transparent::<u32, WP, W>(src)
                        } else {
                            // The source's line is one whole word and this stage
                            // is narrower: unpack each word across several lines.
                            self.scan_words::<W>(src)
                        }
                    }
                    Packing::Plain => {
                        panic!("Memory::fill_from: a quantized source is never plain")
                    }
                },
            }
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
