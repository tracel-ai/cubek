//! Choosing the transport that moves one memory tile's cells into another's, and the copy.
//! Cooperative fills assume every unit their destination names ([`FillUnits`]) runs them: the
//! cube's, or the plane's that owns the stage.

use cubecl::prelude::*;

use super::cooperative::fill_extent;
use crate::*;

/// One side of a fill, as far as choosing a transport goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StoreForm {
    /// How the values sit in the buffer.
    pub(crate) packing: Packing,
    /// The physical line width.
    pub(crate) width: usize,
    /// Whether the coordinates gather.
    pub(crate) gathered: bool,
    /// Whether a read past the logical bound masks to zero.
    pub(crate) masks: bool,
    /// Whether the store has an address (a buffer, not an erased call).
    pub(crate) addressed: bool,
}

/// Which transport moves a memory tile's cells into another memory tile's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TransportKind {
    /// The packed words verbatim.
    Packed,
    /// Every line in the destination's physical order; only for a whole, unmasked, addressed
    /// destination that replaces.
    Straight,
    /// Source and destination each read as a flat run of its own window.
    Scanned(Scan),
}

/// What a [`Scanned`](TransportKind::Scanned) transport reads out of the source per line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scan {
    /// The source's served element, line for line.
    Element,
    /// `u32` words, unpacked as they are read.
    Words,
}

impl TransportKind {
    /// The transport `src` reaches `dst` by; panics on a pairing with no transport.
    pub(crate) fn new(dst: StoreForm, src: StoreForm, access: &Access, space: &Space) -> Self {
        if dst.packing != Packing::Plain {
            assert!(
                access.whole
                    && !access.overhang.masks()
                    && src.packing == dst.packing
                    && dst.width == src.width,
                "TransportKind: a packed stage is a fresh whole buffer filled from the window it \
                 was shaped over, at the same packing and width"
            );
            return TransportKind::Packed;
        }
        if access.whole
            && !access.overhang.masks()
            && src.packing == Packing::Plain
            && access.write == Write::Replace
            && dst.addressed
        {
            // A padded stage assembles each destination line from scalar source cells.
            assert!(
                dst.width == src.width || src.width == 1,
                "TransportKind: a padded stage is filled from a plain scalar operand, but its \
                 source serves {}-wide lines",
                src.width
            );
            return TransportKind::Straight;
        }
        TransportKind::Scanned(Scan::new(dst, src, space))
    }
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// Cooperative cyclic copy of `src` into `self`, whole lines at `self`'s width.
    /// The caller must `sync_cube`, or `sync_plane` for a stage a plane owns, between this fill
    /// and its readers.
    pub(crate) fn fill_from(&mut self, src: &Memory<T>, #[comptime] space: Space) {
        let size!(W) = comptime!(self.store.vector_size);
        // A gathered stage records where it read from: nothing in the staged window says which
        // taps took the boundary value.
        #[comptime]
        match &mut self.source_window {
            ComptimeOption::Some(source) => {
                source.origin = src.window.origin.clone();
                source.bound = src.window.bound.clone();
            }
            ComptimeOption::None => {}
        }
        let addressed = self.store.has_address();
        let src_addressed = src.store.has_address();
        let transport = comptime!(TransportKind::new(
            StoreForm {
                packing: self.store.packing,
                width: self.store.vector_size,
                gathered: !self.projection.is_direct(),
                masks: self.access.overhang.masks(),
                addressed,
            },
            StoreForm {
                packing: src.store.packing,
                width: src.store.vector_size,
                gathered: !src.projection.is_direct(),
                masks: src.access.overhang.masks(),
                addressed: src_addressed,
            },
            &self.access,
            &space
        ));
        match comptime!(transport) {
            TransportKind::Packed => {
                let size!(WP) = comptime!(self.store.packing.physical(self.store.vector_size));
                self.fill_straight::<u32, WP>(src, comptime!(space.clone()));
            }
            TransportKind::Straight => self.fill_straight::<T, W>(src, comptime!(space.clone())),
            TransportKind::Scanned(scan) => self.fill_scanned::<W>(src, comptime!(scan)),
        }
    }
}

impl Scan {
    /// What the scan reads per line, decided by the source's form.
    fn new(dst: StoreForm, src: StoreForm, space: &Space) -> Self {
        assert!(
            !src.gathered && !dst.gathered,
            "TransportKind: a gathered tile fills only a whole, unmasked, unpacked \
             destination (a stage)"
        );
        assert!(
            dst.width == src.width,
            "TransportKind: a plain source is scanned at the destination's width, so a padded \
             stage has to take the straight fill"
        );
        // `storage_extents` rounds the innermost extent up; the scan would miss an unfillable line.
        fill_extent(space, src.width, dst.width, src.masks);
        match src.packing {
            Packing::Plain => Scan::Element,
            Packing::Packed { field: _ } => Scan::Words,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::quant::scheme::QuantValue;

    const K: Axis = Axis(0);
    const N: Axis = Axis(1);

    fn space() -> Space {
        Space::new(&[(K, 8), (N, 32)])
    }

    fn access() -> Access {
        Access {
            whole: true,
            overhang: Overhang::Never,
            write: Write::Replace,
            fill: FillUnits::cube(64),
            storage: Storage::Contiguous,
        }
    }

    fn plain(width: usize) -> StoreForm {
        StoreForm {
            packing: Packing::Plain,
            width,
            gathered: false,
            masks: false,
            addressed: true,
        }
    }

    fn words(width: usize) -> StoreForm {
        StoreForm {
            packing: Packing::Packed {
                field: Field::Quant(QuantValue::Q4F),
            },
            ..plain(width)
        }
    }

    fn kind(dst: StoreForm, src: StoreForm, access: &Access) -> TransportKind {
        TransportKind::new(dst, src, access, &space())
    }

    #[test]
    fn a_whole_plain_stage_is_filled_straight() {
        assert_eq!(kind(plain(4), plain(4), &access()), TransportKind::Straight);
    }

    #[test]
    fn a_padded_stage_is_still_straight() {
        assert_eq!(kind(plain(4), plain(1), &access()), TransportKind::Straight);
    }

    #[test]
    fn a_windowed_masked_or_folding_destination_scans() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        let masked = Access {
            overhang: Overhang::Masked,
            ..access()
        };
        let folding = Access {
            write: Write::Accumulate,
            ..access()
        };
        for access in [windowed, masked, folding] {
            assert_eq!(
                kind(plain(4), plain(4), &access),
                TransportKind::Scanned(Scan::Element),
                "{access:?}"
            );
        }
    }

    #[test]
    fn a_whole_sink_scans() {
        let sink = StoreForm {
            addressed: false,
            ..plain(4)
        };
        assert_eq!(
            kind(sink, plain(4), &access()),
            TransportKind::Scanned(Scan::Element)
        );
    }

    #[test]
    fn a_packed_stage_moves_words_and_nothing_else() {
        assert_eq!(kind(words(8), words(8), &access()), TransportKind::Packed);
    }

    #[test]
    fn the_source_names_what_a_scan_reads() {
        let scanned = Access {
            whole: false,
            ..access()
        };
        for (src, scan) in [(plain(4), Scan::Element), (words(4), Scan::Words)] {
            assert_eq!(
                kind(plain(4), src, &scanned),
                TransportKind::Scanned(scan),
                "{src:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "a gathered tile fills only a whole")]
    fn a_gathered_source_refuses_a_scan() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        let gathered = StoreForm {
            gathered: true,
            ..plain(4)
        };
        kind(plain(4), gathered, &windowed);
    }
}
