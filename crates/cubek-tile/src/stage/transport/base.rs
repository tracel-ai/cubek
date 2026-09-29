//! Which transport moves one memory tile's cells into another's, and the copy that runs it.
//!
//! Every cooperative fill under this module distributes its lines out over `CUBE_DIM` workers indexed
//! by `UNIT_POS`, assuming every unit of the cube runs it. A walk that sets planes aside to fill
//! its stages ([`Levels::filled_by`](crate::Levels::filled_by)) breaks that: a wrong answer, not a
//! hang. Such a walk is refused where the two meet, and only a bulk copy may be filled by planes
//! of their own.

use cubecl::prelude::*;

use super::cooperative::fill_extent;
use crate::*;

/// One side of a fill, as far as choosing a transport goes: how its values sit in its buffer, the
/// line it is served in, and whether its coordinates gather.
///
/// Read off a [`Memory`] at comptime and nothing else, so [`TransportKind::new`] is plain data in
/// and one named case out, testable without a device.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StoreForm {
    /// How the values sit in the buffer: as served, or several to a `u32` word.
    pub(crate) packing: Packing,
    /// The physical line the store is served in.
    pub(crate) width: usize,
    /// Whether the coordinates gather: a compacted window rather than a box.
    pub(crate) gathered: bool,
    /// Whether a read past the logical bound masks to zero.
    pub(crate) masks: bool,
    /// Whether the store has an address: a buffer, rather than an erased call that only stores
    /// or loads through its layout.
    pub(crate) addressed: bool,
    /// Who moves the store's lines when it is the source of a fill ([`Access::delivery`]).
    pub(crate) delivery: Delivery,
}

/// Which transport moves a memory tile's cells into another memory tile's.
///
/// The two stores and the destination's access decide it once, at comptime, so the fill itself is
/// a table of three well-named arms instead of the conditions that pick between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TransportKind {
    /// The packed words verbatim, and nothing beside them: scales are an operand of their own,
    /// and a decode is a copy the kernel states ([`Tile::mul`]).
    Packed,
    /// Every line in the destination's own physical order, the source decoded once per line. The
    /// write is linear, which is why only a whole, unmasked destination that replaces takes it,
    /// and only one with an address: a sink is written through its layout, so it scans.
    Straight,
    /// Source and destination each read as a flat run of its own window, which pairs them only
    /// where the two are the same box. What the source hands out per line names the read.
    Scanned(Scan),
}

/// What a [`Scanned`](TransportKind::Scanned) transport reads out of the source per line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scan {
    /// The source's served element, one line for one line.
    Element,
    /// `u32` words, one for one, their values unpacked as they are read.
    Words,
}

impl TransportKind {
    /// The transport `src` reaches `dst` by, or a panic naming the pairing that has no transport.
    ///
    /// Every claim a fill rests on is asserted here rather than at the leaf that would trip over
    /// it: what a packed stage owes its source, what a straight fill owes the widths,
    /// what the innermost extent owes both ([`fill_extent`]), and what an async copy owes the
    /// transport it rides ([`refuse_async`](TransportKind::refuse_async)).
    pub(crate) fn new(dst: StoreForm, src: StoreForm, access: &Access, space: &Space) -> Self {
        let kind = Self::choose(dst, src, access, space);
        match src.delivery {
            Delivery::AsyncPerUnit => kind.refuse_async(dst, src),
            Delivery::AsyncBulk => panic!(
                "TransportKind: a bulk copy moves a stage that is one contiguous run in its \
                 source's byte order, and no fill checks that yet"
            ),
            Delivery::SyncPerUnit | Delivery::Tma | Delivery::Procedural => {}
        }
        kind
    }

    /// An async copy moves a source line's bytes to one destination line, by address, without a
    /// unit touching them: only a transport whose every line is such a move takes one. A scan
    /// may fold or window its destination, and a padded stage assembles each line out of
    /// several source cells.
    fn refuse_async(self, dst: StoreForm, src: StoreForm) {
        assert!(
            !matches!(self, TransportKind::Scanned(_)),
            "TransportKind: an async copy fills a whole, unmasked stage that replaces, and this \
             destination is windowed, masked, folding or a sink"
        );
        assert!(
            dst.width == src.width,
            "TransportKind: an async copy moves whole source lines, but this stage is served in \
             {}-wide lines out of {}-wide ones",
            dst.width,
            src.width
        );
        assert!(
            src.addressed,
            "TransportKind: an async copy reads its source by address, and this one is an erased call"
        );
    }

    /// The transport the two forms and the destination's access leave, before any claim a
    /// delivery adds.
    fn choose(dst: StoreForm, src: StoreForm, access: &Access, space: &Space) -> Self {
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
            // A padded stage is served in lines its source cannot hand out whole, so each
            // destination line is assembled unit by unit out of scalar source cells.
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
    /// Memory transport leaf: cooperative cyclic copy of `src` into `self`, whole
    /// `Vector<T, W>` lines at `self`'s width, unit `u` moving lines `u`, `u + CUBE_DIM`, ….
    /// The caller owns the rendezvous: a `sync_cube` must separate this fill from its readers.
    ///
    /// `space` is the logical space both sides carry. A gathered `src` stages the compacted
    /// *window* rather than its logical tile, so the fill stays a box copy and the gather stays at
    /// the leaf's read; see [`fill_straight`](Memory::fill_straight) and [`Compaction`].
    pub(crate) fn fill_from(&mut self, src: &Memory<T>, #[comptime] space: Space) {
        let size!(W) = comptime!(self.store.vector_size);
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
        let addressed = self.store.has_address();
        let src_addressed = src.store.has_address();
        let transport = comptime!(TransportKind::new(
            StoreForm {
                packing: self.store.packing,
                width: self.store.vector_size,
                gathered: !self.projection.is_direct(),
                masks: self.access.overhang.masks(),
                addressed,
                delivery: self.access.delivery,
            },
            StoreForm {
                packing: src.store.packing,
                width: src.store.vector_size,
                gathered: !src.projection.is_direct(),
                masks: src.access.overhang.masks(),
                addressed: src_addressed,
                delivery: src.access.delivery,
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
    /// What the scan reads per line, which the source's own form decides: the destination is
    /// plain here, since every other destination took a transport above.
    fn new(dst: StoreForm, src: StoreForm, space: &Space) -> Self {
        assert!(
            !src.gathered && !dst.gathered,
            "TransportKind: a gathered tile fills only a whole, unmasked, unpacked \
             destination (a stage)"
        );
        // The read unpacks what the source stores: its element as it is, or the values its words
        // hold. A decode beyond that is the kernel's statement ([`Tile::copy_from`] a scaled
        // source), which never reaches this transport.
        assert!(
            dst.width == src.width,
            "TransportKind: a plain source is scanned at the destination's width, so a padded \
             stage has to take the straight fill"
        );
        // Equal widths here, so this only asks that the innermost extent is whole lines:
        // `storage_extents` rounds it up, and nothing on the scan path would otherwise notice the
        // last line the source cannot fill.
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

    /// A box whose innermost extent is whole lines at every width these tests use.
    fn space() -> Space {
        Space::new(&[(K, 8), (N, 32)])
    }

    /// A fresh whole stage: what every destination below starts from.
    fn access() -> Access {
        Access {
            whole: true,
            overhang: Overhang::Never,
            write: Write::Replace,
            units: 64,
            storage: Storage::Contiguous,
            delivery: Delivery::SyncPerUnit,
        }
    }

    fn plain(width: usize) -> StoreForm {
        StoreForm {
            packing: Packing::Plain,
            width,
            gathered: false,
            masks: false,
            addressed: true,
            delivery: Delivery::SyncPerUnit,
        }
    }

    fn asynchronous(form: StoreForm) -> StoreForm {
        StoreForm {
            delivery: Delivery::AsyncPerUnit,
            ..form
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

    /// The ordinary stage: a whole plain destination that replaces is written in its own physical
    /// order, one line at a time.
    #[test]
    fn a_whole_plain_stage_is_filled_straight() {
        assert_eq!(kind(plain(4), plain(4), &access()), TransportKind::Straight);
    }

    /// A padded stage is served in lines its source cannot hand out whole, and still fills
    /// straight: the assembly is the leaf's, not another transport.
    #[test]
    fn a_padded_stage_is_still_straight() {
        assert_eq!(kind(plain(4), plain(1), &access()), TransportKind::Straight);
    }

    /// Three things take the destination off the straight path, each on its own: a window that no
    /// longer covers the buffer, a masked edge, and a store that folds.
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

    /// A whole destination with no address, a sink, is written through its layout: the straight
    /// fill's linear write needs the buffer a call does not have.
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

    /// A packed stage takes its words verbatim and nothing beside them.
    #[test]
    fn a_packed_stage_moves_words_and_nothing_else() {
        assert_eq!(kind(words(8), words(8), &access()), TransportKind::Packed);
    }

    /// The source's own form names the scan: its element, or whole words.
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

    /// An async copy rides the transports whose every line is one source line moved whole: the
    /// straight and packed ones, never a scan.
    #[test]
    fn an_async_copy_rides_a_straight_or_packed_fill() {
        assert_eq!(
            kind(plain(4), asynchronous(plain(4)), &access()),
            TransportKind::Straight
        );
        assert_eq!(
            kind(words(8), asynchronous(words(8)), &access()),
            TransportKind::Packed
        );
    }

    #[test]
    #[should_panic(expected = "an async copy fills a whole, unmasked stage")]
    fn an_async_copy_refuses_a_scan() {
        let masked = Access {
            overhang: Overhang::Masked,
            ..access()
        };
        kind(plain(4), asynchronous(plain(4)), &masked);
    }

    /// A padded stage assembles each line out of scalar cells, which no single copy moves.
    #[test]
    #[should_panic(expected = "an async copy moves whole source lines")]
    fn an_async_copy_refuses_a_padded_stage() {
        kind(plain(4), asynchronous(plain(1)), &access());
    }

    /// A bulk copy moves the stage as one run, which only a source laid out byte for byte like the
    /// stage allows, and nothing checks that yet.
    #[test]
    #[should_panic(expected = "a bulk copy moves a stage that is one contiguous run")]
    fn a_bulk_copy_is_refused_until_its_layout_is_checked() {
        let bulk = StoreForm {
            delivery: Delivery::AsyncBulk,
            ..plain(4)
        };
        kind(plain(4), bulk, &access());
    }

    /// A gather stages its compacted window, which only a whole plain stage can hold.
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
