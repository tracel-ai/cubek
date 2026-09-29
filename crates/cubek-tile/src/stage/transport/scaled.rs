//! The decoding transport: a source carrying scales ([`Tile::mul`]) or indexing a table
//! ([`Tile::lookup`]) copied into a destination that holds plain values, each line decoded as it
//! lands.
//!
//! This is where a kernel decodes on purpose: `stage.copy_from(&w.lookup(&t).mul(&scales))` states
//! the decode at the copy, and nothing decodes behind a read the kernel did not write.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// [`copy_from`](Tile::copy_from) a scaled `src`: each unit of the cube takes the lines `u`,
    /// `u + CUBE_DIM`, … of the window, reads each through the source's packed view (so packed
    /// values unpack), multiplies it by every level of the source's scales at the line's first
    /// value, and writes it where it lies in `self`.
    ///
    /// **A line reads one scale**, the innermost level's at its first value, as every scaled read
    /// does: a block of scales spans whole lines of the values it covers.
    ///
    /// **The two span one box.** The source may carry axes the destination does not, each one
    /// wide (a batch or head the cube already fixed); every other axis is the destination's, in
    /// its order and at its extent.
    pub(crate) fn copy_scaled_from(&mut self, src: &Tile<T>) {
        let delivery = src.delivery();
        comptime!(assert!(
            !delivery.is_async(),
            "Tile::copy_from: a decoding copy decodes each line in the unit that loads it, and \
             this source is delivered {delivery:?}, which lands its bytes as they lie; deliver it \
             SyncPerUnit, or stage it as it lies and decode out of the stage"
        ));
        let space = comptime!(src.place.space.clone());
        comptime!(kept_axes(&space, &self.place.space));
        comptime!(assert!(
            space.axes().all(|axis| !space.is_dynamic(axis)),
            "Tile::copy_from: a decoding copy reads its window through the values' own axes, which \
             needs their extents at expansion; this window spans a dynamic axis. Copy a region a \
             level cuts to (a stage, a leaf) rather than the whole dynamic tile"
        ));
        // The source is read a whole line at a time (a packed source's line is a whole word), and
        // each line is written as the destination's lines it holds: several where the destination
        // is served narrower than a word.
        let vw = self.vector_size();
        let sw = src.vector_size();
        comptime!(assert!(
            sw.is_multiple_of(vw),
            "Tile::copy_from: a decoding copy writes each source line as whole destination lines, \
             and a {sw}-wide source line does not split into {vw}-wide ones"
        ));
        let decode = src.mem("Tile::copy_from");
        // A packed source whose destination is served narrower than its line is read as the words
        // it lies in and unpacked a destination line at a time, so no line wider than the
        // destination's is ever held: a device's vectors may be narrower than a word.
        match comptime!(decode.store.packing) {
            Packing::Packed { field } if vw < sw => self.copy_decoded_words(src, field),
            _ => self.copy_decoded_lines(src),
        }
    }

    /// [`copy_scaled_from`](Tile::copy_scaled_from) reading the source a whole line at a time and
    /// splitting it into the destination's lines.
    fn copy_decoded_lines(&mut self, src: &Tile<T>) {
        let space = comptime!(src.place.space.clone());
        let kept = comptime!(kept_axes(&space, &self.place.space));
        let (vw, sw) = (self.vector_size(), src.vector_size());
        let size!(VW) = vw;
        let size!(SW) = sw;
        let rank = comptime!(space.rank());
        let decode = src.mem("Tile::copy_from");
        let looked_up = src.looked_up();
        let values = src.nd_packed::<SW>(comptime!(Guard::Checked));
        let mut out = self.nd_mut::<VW>();
        let shape = values.shape();
        for line in range_stepped(UNIT_POS, line_count(&shape, rank), CUBE_DIM) {
            let digits = line_digits(line, &shape, rank);
            let whole = values.read(digits_at(&digits, rank));
            #[unroll]
            for c in 0..comptime!(sw / vw) {
                let mut chunk = Vector::<T, VW>::empty();
                #[unroll]
                for j in 0..vw {
                    chunk.insert(j, whole.extract(comptime!(c * vw) + j));
                }
                land_chunk::<T, VW>(
                    &mut out,
                    decode,
                    &digits,
                    chunk,
                    c,
                    sw,
                    space.clone(),
                    kept.clone(),
                    looked_up,
                );
            }
        }
    }

    /// [`copy_scaled_from`](Tile::copy_scaled_from) reading a packed source's words and unpacking
    /// each destination line's fields straight out of the word that holds them.
    fn copy_decoded_words(&mut self, src: &Tile<T>, #[comptime] field: Field) {
        let space = comptime!(src.place.space.clone());
        let kept = comptime!(kept_axes(&space, &self.place.space));
        let (vw, sw) = (self.vector_size(), src.vector_size());
        let size!(VW) = vw;
        let rank = comptime!(space.rank());
        let decode = src.mem("Tile::copy_from");
        let looked_up = src.looked_up();
        let physical = comptime!(decode.store.packing.physical(sw));
        let size!(WP) = physical;
        let layout = decode.axis_projection(comptime!(space.clone()));
        let words = decode.nd_words::<WP>(layout, comptime!(Guard::Checked));
        let mut out = self.nd_mut::<VW>();
        let shape = words.shape();
        let (bits, per_word) = comptime!((field.size_bits(), field.per_word()));
        for line in range_stepped(UNIT_POS, line_count(&shape, rank), CUBE_DIM) {
            let digits = line_digits(line, &shape, rank);
            let line_words = words.read(digits_at(&digits, rank));
            #[unroll]
            for c in 0..comptime!(sw / vw) {
                let (word, shift) = comptime!(((c * vw) / per_word, ((c * vw) % per_word) * bits));
                let mut one = Vector::<u32, Const<1>>::empty();
                one.insert(0usize, line_words.extract(word) >> comptime!(shift as u32));
                let chunk = unpack_line::<T, Const<1>, VW>(one, field);
                land_chunk::<T, VW>(
                    &mut out,
                    decode,
                    &digits,
                    chunk,
                    c,
                    sw,
                    space.clone(),
                    kept.clone(),
                    looked_up,
                );
            }
        }
    }
}

/// Which of `src`'s axes `dst` spans, positionally: the destination is the source with its
/// one-wide extra axes dropped, the rest in the same order at the same extents, the innermost
/// shared.
fn kept_axes(src: &Space, dst: &Space) -> Vec<bool> {
    let kept: Vec<bool> = src
        .axes()
        .map(|axis| dst.axes().any(|a| a == axis))
        .collect();
    let shared: Vec<Axis> = src
        .axes()
        .filter(|&axis| dst.axes().any(|a| a == axis))
        .collect();
    let mismatch = || {
        format!(
            "Tile::copy_from: a decoding copy writes the box it reads, and {src:?} does not \
             narrow to {dst:?} by dropping one-wide axes"
        )
    };
    assert!(shared == dst.axes().collect::<Vec<_>>(), "{}", mismatch());
    assert!(kept.last() == Some(&true), "{}", mismatch());
    for axis in src.axes() {
        match kept[src.position(axis)] {
            true => assert!(src.extent(axis) == dst.extent(axis), "{}", mismatch()),
            false => assert!(src.extent(axis) == 1, "{}", mismatch()),
        }
    }
    kept
}

/// Lines in a view of `shape`, its innermost counted in lines.
#[cube]
fn line_count(shape: &CoordsDyn, #[comptime] rank: usize) -> u32 {
    let mut lines = 1u32;
    #[unroll]
    for p in 0..rank {
        lines *= *shape.index(p);
    }
    lines
}

/// The `line`-th line's digits under `shape`, innermost fastest.
#[cube]
fn line_digits(line: u32, shape: &CoordsDyn, #[comptime] rank: usize) -> Array<u32> {
    let mut digits = Array::<u32>::new(comptime!(rank));
    let mut rest = line;
    #[unroll]
    for i in 0..rank {
        let p = comptime!(rank - 1 - i);
        let extent = *shape.index(p);
        digits[p] = rest % extent;
        rest /= extent;
    }
    digits
}

/// `digits` as a view addresses them.
#[cube]
fn digits_at(digits: &Array<u32>, #[comptime] rank: usize) -> CoordsDyn {
    let mut at = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        at.push(digits[p]);
    }
    at
}

/// Decode the `c`-th destination line of the source line at `digits` (`sw` values wide) and
/// write it: every index replaced by its table entry, the line scaled at its first value.
#[cube]
fn land_chunk<T: Numeric, VW: Size>(
    out: &mut MaskedMut<'_, Vector<T, VW>, CoordsDyn>,
    decode: &Memory<T>,
    digits: &Array<u32>,
    stored: Vector<T, VW>,
    #[comptime] c: usize,
    #[comptime] sw: usize,
    #[comptime] space: Space,
    #[comptime] kept: Vec<bool>,
    #[comptime] looked_up: bool,
) {
    let vw = VW::value();
    let rank = comptime!(space.rank());
    // An index names its table entry before anything scales it.
    let decoded = if comptime!(looked_up) {
        let mut entries = Vector::<T, VW>::empty();
        #[unroll]
        for j in 0..vw {
            let index = u32::cast_from(stored.extract(j));
            entries.insert(j, T::cast_from(decode.codebook.entry(index)));
        }
        entries
    } else {
        stored
    };
    // The chunk's first value: where it is written (its innermost counted in the destination's
    // lines) and where its scales are looked up.
    let first = digits[comptime!(rank - 1)] * comptime!(sw as u32) + comptime!((c * vw) as u32);
    let mut write_at = CoordsDyn::new();
    let mut coords = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            write_at.push(first / comptime!(vw as u32));
            coords.push(first);
        } else {
            if comptime!(kept[p]) {
                write_at.push(digits[p]);
            }
            coords.push(digits[p]);
        }
    }
    let scale = decode.factor.at_coords(&coords, space);
    out.write(write_at, decoded * Vector::<T, VW>::cast_from(scale));
}
