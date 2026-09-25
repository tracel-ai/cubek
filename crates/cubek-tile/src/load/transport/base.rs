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

/// One side of a fill, as far as choosing a transport goes: whether it carries a scheme, how its
/// values sit in its buffer, the line it is served in, and whether its coordinates gather.
///
/// Read off a [`Memory`] at comptime and nothing else, so [`TransportKind::new`] is plain data in
/// and one named case out, testable without a device.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StoreForm {
    /// Whether the store carries a quantization scheme, so its cells mean something only once
    /// their scales are folded in.
    pub(crate) scheme: bool,
    /// How the values sit in the buffer: as served, as `i8` codes, or several to a `u32` word.
    pub(crate) packing: Packing,
    /// The physical line the store is served in.
    pub(crate) width: usize,
    /// Whether the coordinates gather: a compacted window rather than a box.
    pub(crate) gathered: bool,
    /// Whether a read past the logical bound masks to zero.
    pub(crate) masks: bool,
}

/// Which transport moves a memory tile's cells into another memory tile's.
///
/// The two stores and the destination's access decide it once, at comptime, so the fill itself is
/// a table of four well-named arms instead of the conditions that pick between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TransportKind {
    /// The packed words verbatim in the destination's own physical order, then the scales beside
    /// them, so a leaf read dequantizes straight out of shared memory with no `f32` inflation.
    /// The variant names the packing the words are stored at.
    Quantized(Packing),
    /// The packed words verbatim, and nothing beside them: the scales are an operand of their own.
    Packed,
    /// Every line in the destination's own physical order, the source decoded once per line. The
    /// write is linear, which is why only a whole, unmasked destination that replaces takes it.
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
    /// `i8` codes under a scheme, one line for one line.
    Codes,
    /// One `u32` word spread over several of the destination's narrower lines, which is how a
    /// packed operand stages on a device whose vectors cannot cover a word.
    SubWord,
}

impl TransportKind {
    /// The transport `src` reaches `dst` by, or a panic naming the pairing that has no transport.
    ///
    /// Every claim a fill rests on is asserted here rather than at the leaf that would trip over
    /// it: what a quantized or packed stage owes its source, what a straight fill owes the widths,
    /// and what the innermost extent owes both ([`fill_extent`]).
    pub(crate) fn new(dst: StoreForm, src: StoreForm, access: &Access, space: &Space) -> Self {
        if dst.scheme {
            // Unreachable in practice: `Memory::global` already asserts `quant.is_none() ||
            // coords.is_direct()` at construction, so a gathered `src` never carries a quantized
            // form to begin with. Kept as defense in case that invariant ever loosens.
            assert!(
                !src.gathered,
                "TransportKind: a gathered operand cannot stage in its quantized form"
            );
            assert!(
                access.whole && !access.overhang.masks(),
                "TransportKind: a quantized stage is always a fresh whole buffer"
            );
            assert!(
                src.scheme,
                "TransportKind: a quantized stage must be filled from a quantized source"
            );
            // The stage was allocated at the scheme's own storage ([`scheme_packing`]): one
            // element per value, or `u32` words carrying several, the physical line then that
            // much narrower than the served one.
            assert!(
                dst.packing != Packing::Plain,
                "TransportKind: a quantized stage is never plain"
            );
            return TransportKind::Quantized(dst.packing);
        }
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
        {
            // A padded stage is served in lines its source cannot hand out whole, so each
            // destination line is assembled lane by lane out of scalar source cells.
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
        let transport = comptime!(TransportKind::new(
            StoreForm {
                scheme: self.store.quant.is_some(),
                packing: self.store.packing,
                width: self.store.vector_size,
                gathered: !self.projection.is_direct(),
                masks: self.access.overhang.masks(),
            },
            StoreForm {
                scheme: src.store.quant.is_some(),
                packing: src.store.packing,
                width: src.store.vector_size,
                gathered: !src.projection.is_direct(),
                masks: src.access.overhang.masks(),
            },
            &self.access,
            &space
        ));
        match comptime!(transport) {
            TransportKind::Quantized(packing) => {
                let size!(WP) = comptime!(packing.physical(self.store.vector_size));
                match comptime!(packing) {
                    Packing::Native => self.fill_straight::<i8, WP>(src, comptime!(space.clone())),
                    Packing::Packed { field: _ } => {
                        self.fill_straight::<u32, WP>(src, comptime!(space.clone()))
                    }
                    Packing::Plain => comptime!(unreachable!()),
                }
                self.stage_scales(src);
            }
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
            "TransportKind: a gathered tile fills only a whole, unmasked, unquantized \
             destination (a stage)"
        );
        // The read decodes at the source's true storage element: `T` for a plain tile, else the
        // quantized store's element recovered from its scheme. That lets a plain `copy_from`
        // dequantize on its own; the kernel never threads the storage element.
        match (src.scheme, src.packing) {
            (false, packing) => {
                assert!(
                    dst.width == src.width,
                    "TransportKind: a plain source is scanned at the destination's width, so a \
                     padded stage has to take the straight fill"
                );
                // Equal widths here, so this only asks that the innermost extent is whole lines:
                // `storage_extents` rounds it up, and nothing on the scan path would otherwise
                // notice the last line the source cannot fill.
                fill_extent(space, src.width, dst.width, src.masks);
                match packing {
                    Packing::Plain => Scan::Element,
                    Packing::Packed { field: _ } => Scan::Words,
                    Packing::Native => panic!(
                        "TransportKind: a native store with nothing to fold in serves its own \
                         element; bind it as that element"
                    ),
                }
            }
            (true, Packing::Native) => Scan::Codes,
            (true, Packing::Packed { field: _ }) if src.width == dst.width => Scan::Words,
            (true, Packing::Packed { field: _ }) => Scan::SubWord,
            (true, Packing::Plain) => {
                panic!("TransportKind: a quantized source is never plain")
            }
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
        }
    }

    fn plain(width: usize) -> StoreForm {
        StoreForm {
            scheme: false,
            packing: Packing::Plain,
            width,
            gathered: false,
            masks: false,
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

    fn quantized(width: usize) -> StoreForm {
        StoreForm {
            scheme: true,
            ..words(width)
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

    /// A stage carrying the scheme stages the words verbatim and the scales beside them, and the
    /// variant names the packing it stores them at.
    #[test]
    fn a_quantized_stage_carries_its_scales() {
        assert_eq!(
            kind(quantized(8), quantized(8), &access()),
            TransportKind::Quantized(Packing::Packed {
                field: Field::Quant(QuantValue::Q4F)
            })
        );
    }

    /// A packed stage with no scheme takes the same verbatim copy and nothing beside it.
    #[test]
    fn a_packed_stage_moves_words_and_nothing_else() {
        assert_eq!(kind(words(8), words(8), &access()), TransportKind::Packed);
    }

    /// The source's own form names the scan: its element, whole words, codes under a scheme, or
    /// one word spread over several narrower lines.
    #[test]
    fn the_source_names_what_a_scan_reads() {
        let scanned = Access {
            whole: false,
            ..access()
        };
        let native = StoreForm {
            packing: Packing::Native,
            ..quantized(4)
        };
        for (src, scan) in [
            (plain(4), Scan::Element),
            (words(4), Scan::Words),
            (native, Scan::Codes),
            (quantized(4), Scan::Words),
            (quantized(8), Scan::SubWord),
        ] {
            assert_eq!(
                kind(plain(4), src, &scanned),
                TransportKind::Scanned(scan),
                "{src:?}"
            );
        }
    }

    /// A quantized stage is a fresh whole buffer: a window onto one has no transport.
    #[test]
    #[should_panic(expected = "a quantized stage is always a fresh whole buffer")]
    fn a_windowed_quantized_stage_is_refused() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        kind(quantized(8), quantized(8), &windowed);
    }

    /// The scales have to come from somewhere: a plain source cannot fill a quantized stage.
    #[test]
    #[should_panic(expected = "must be filled from a quantized source")]
    fn a_quantized_stage_refuses_a_plain_source() {
        kind(quantized(8), plain(8), &access());
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
