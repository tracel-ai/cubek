//! Which transport moves one memory tile's cells into another's, and the copy that runs it.
//!
//! Every cooperative fill under this module deals its lines out over `CUBE_DIM` workers indexed
//! by `UNIT_POS`, assuming every unit of the cube runs it. A walk that sets planes aside to fill
//! its stages ([`Level::filled_by`](crate::Level::filled_by)) breaks that: a wrong answer, not a
//! hang. Such a walk is refused where the two meet, and only a bulk copy may be filled by planes
//! of their own.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// Memory transport leaf: cooperative cyclic copy of `src` into `self`, whole
    /// `Vector<T, W>` lines at `self`'s width, unit `u` moving lines `u`, `u + CUBE_DIM`, ….
    /// The caller owns the rendezvous: a `sync_cube` must separate this fill from its readers.
    ///
    /// `space` is the logical space both sides carry. A gathered `src` stages the compacted
    /// *window* rather than its logical tile, so the fill stays a box copy and the gather stays at
    /// the leaf's read; see [`fill_straight`](Memory::fill_straight) and [`Compaction`].
    pub(crate) fn fill_from(&mut self, src: &Memory<T>, #[comptime] space: Space) {
        let size!(W) = comptime!(self.store.vector_size);
        let gathered = comptime!(!src.projection.is_direct());
        // A gathered stage keeps where it read from: the copy below writes the boundary's value
        // for every tap outside `src`, and nothing in the staged window says which those were.
        #[comptime]
        match &mut self.source_window {
            ComptimeOption::Some(source) => {
                source.origin = src.window.origin.clone();
                source.bound = src.window.bound.clone();
            }
            ComptimeOption::None => {}
        }
        if comptime!(self.store.quant.is_some()) {
            // Unreachable in practice: `Memory::global` already asserts `quant.is_none() ||
            // coords.is_direct()` at construction, so a gathered `src` never carries a quantized
            // form to begin with. Kept as defense in case that invariant ever loosens.
            comptime!(assert!(
                !gathered,
                "Memory::fill_from: a gathered operand cannot stage in its quantized form"
            ));
            // Quant → quant: stage the packed storage words verbatim through the straight-line
            // fill, then the scales beside them, so a leaf read dequantizes straight out of smem
            // with no f32 inflation. A quantized stage is a fresh whole buffer; nothing masks.
            comptime!(assert!(
                self.access.whole && !self.access.overhang.masks(),
                "Memory::fill_from: a quantized stage is always a fresh whole buffer"
            ));
            comptime!(assert!(
                src.store.quant.is_some(),
                "Memory::fill_from: a quantized stage must be filled from a quantized source"
            ));
            // The stage was allocated at the scheme's own storage ([`scheme_packing`]): one
            // element per value, or `u32` words carrying several, the physical line then that
            // much narrower than the served one.
            let size!(WP) = comptime!(self.store.packing.physical(self.store.vector_size));
            match comptime!(self.store.packing) {
                Packing::Native => self.fill_straight::<i8, WP>(src, comptime!(space.clone())),
                Packing::Packed { field: _ } => {
                    self.fill_straight::<u32, WP>(src, comptime!(space.clone()))
                }
                Packing::Plain => panic!("Memory::fill_from: a quantized stage is never plain"),
            }
            self.stage_scales(src);
        } else if comptime!(self.store.packing != Packing::Plain) {
            // Packed → packed: the words verbatim, the same straight-line fill a quantized stage
            // takes, with nothing beside them — the scales are an operand of their own.
            comptime!(assert!(
                self.access.whole
                    && !self.access.overhang.masks()
                    && src.store.packing == self.store.packing
                    && self.store.vector_size == src.store.vector_size,
                "Memory::fill_from: a packed stage is a fresh whole buffer filled from the \
                 window it was shaped over, at the same packing and width"
            ));
            let size!(WP) = comptime!(self.store.packing.physical(self.store.vector_size));
            self.fill_straight::<u32, WP>(src, comptime!(space.clone()));
        } else if comptime!(
            self.access.whole
                && !self.access.overhang.masks()
                && src.store.packing == Packing::Plain
                && self.access.write == Write::Replace
        ) {
            // Plain → plain, whole destination that *replaces*: fill in destination-physical order
            // (the write is linear; only the source decodes, once per line). A folding destination
            // has no address to fill directly, so it takes the layout walk below (the sink's call).
            //
            // A padded stage is served in lines its source cannot hand out whole, so each
            // destination line is assembled lane by lane.
            if comptime!(self.store.vector_size != src.store.vector_size) {
                comptime!(assert!(
                    src.store.vector_size == 1,
                    "Memory::fill_from: a padded stage is filled from a plain scalar operand, \
                     but its source serves {}-wide lines",
                    src.store.vector_size
                ));
            }
            self.fill_straight::<T, W>(src, comptime!(space.clone()));
        } else {
            self.fill_scanned::<W>(src, comptime!(space.clone()));
        }
    }
}
