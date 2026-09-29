//! The decoding transport: a source carrying scales ([`Tile::mul`]) or indexing a table
//! ([`Tile::lookup`]) copied into a destination that holds plain values, each line decoded as it
//! lands.
//!
//! This is where a kernel decodes on purpose: `stage.copy_from(&w.lookup(&t).mul(&scales))` states
//! the decode at the copy, and nothing decodes behind a read the kernel did not write.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

/// What every line of one decoding copy shares, settled once at expansion.
#[derive(Clone)]
struct Decode {
    /// The source's space, which its scales are read against.
    space: Space,
    /// Which of the source's axes the destination spans; the rest are one wide.
    kept: Vec<bool>,
    /// The source's line and the destination's.
    sw: usize,
    vw: usize,
    /// Whether each value is an index into the source's table.
    looked_up: bool,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// [`copy_from`](Tile::copy_from) a scaled `src`: the cube's units take the source's lines in
    /// turn, and each lands as the destination lines it holds, looked up and scaled.
    ///
    /// **A destination line reads one scale**, at its first value: a block of scales spans whole
    /// lines of the values it covers.
    ///
    /// **The two span one box.** The source may carry axes the destination does not, each one
    /// wide (a batch or head the cube already fixed); every other axis is the destination's, in
    /// its order and at its extent.
    pub(crate) fn copy_scaled_from(&mut self, src: &Tile<T>) {
        let (sw, vw, looked_up) = (src.vector_size(), self.vector_size(), src.looked_up());
        let decode = comptime!(Decode::of(
            &src.place.space,
            &self.place.space,
            sw,
            vw,
            looked_up,
        ));
        let packing = src.packing();
        match comptime!(packing) {
            Packing::Plain => self.copy_decoded_lines(src, decode),
            Packing::Packed { field } => self.copy_decoded_words(src, field, decode),
        }
    }

    /// A plain source, read a whole line at a time and split into the destination's lines.
    fn copy_decoded_lines(&mut self, src: &Tile<T>, #[comptime] decode: Decode) {
        let size!(SW) = comptime!(decode.sw);
        let size!(VW) = comptime!(decode.vw);
        let values = src.nd_packed::<SW>(comptime!(Guard::Checked));
        let mut out = self.nd_mut::<VW>();
        let extents = comptime!(line_extents(
            &decode.space,
            decode.sw,
            0,
            decode.space.rank()
        ));
        let lines = comptime!(extents.iter().product::<usize>() as u32);
        for line in range_stepped(UNIT_POS, lines, CUBE_DIM) {
            let at = coords_of_line(line, comptime!(extents.clone()), comptime!(decode.sw));
            let whole = values.read(as_dyn(&at, comptime!(decode.sw)));
            #[unroll]
            for c in 0..comptime!(decode.sw / decode.vw) {
                let mut chunk = Vector::<T, VW>::empty();
                #[unroll]
                for j in 0..comptime!(decode.vw) {
                    chunk.insert(j, whole.extract(comptime!(c * decode.vw) + j));
                }
                land::<T, VW>(
                    &mut out,
                    src.mem("Tile::copy_from"),
                    &at,
                    chunk,
                    c,
                    decode.clone(),
                );
            }
        }
    }

    /// A packed source, read as the words its line lies in: each destination line unpacks its
    /// own fields out of the words that hold them, so no line wider than the destination's is ever
    /// held (a device's vectors may be narrower than a word).
    fn copy_decoded_words(
        &mut self,
        src: &Tile<T>,
        #[comptime] field: Field,
        #[comptime] decode: Decode,
    ) {
        let (bits, per_word) = comptime!((field.size_bits(), field.per_word()));
        let size!(WP) = comptime!(decode.sw / per_word);
        let size!(VW) = comptime!(decode.vw);
        // The words one destination line reads: several where it covers more than one word, else
        // the one its fields lie in.
        let nq = comptime!((decode.vw / per_word).max(1));
        let size!(NQ) = nq;
        let mem = src.mem("Tile::copy_from");
        let layout = mem.axis_projection(comptime!(decode.space.clone()));
        let words = mem.nd_words::<WP>(layout, comptime!(Guard::Checked));
        let mut out = self.nd_mut::<VW>();
        let extents = comptime!(line_extents(
            &decode.space,
            decode.sw,
            0,
            decode.space.rank()
        ));
        let lines = comptime!(extents.iter().product::<usize>() as u32);
        for line in range_stepped(UNIT_POS, lines, CUBE_DIM) {
            let at = coords_of_line(line, comptime!(extents.clone()), comptime!(decode.sw));
            let line_words = words.read(as_dyn(&at, comptime!(decode.sw)));
            #[unroll]
            for c in 0..comptime!(decode.sw / decode.vw) {
                let first = comptime!(c * decode.vw);
                let shift = comptime!(((first % per_word) * bits) as u32);
                let mut held = Vector::<u32, NQ>::empty();
                #[unroll]
                for w in 0..nq {
                    held.insert(
                        w,
                        line_words.extract(comptime!(first / per_word) + w) >> shift,
                    );
                }
                let chunk = unpack_line::<T, NQ, VW>(held, field);
                land::<T, VW>(&mut out, mem, &at, chunk, c, decode.clone());
            }
        }
    }
}

impl Decode {
    /// The shared facts of copying `src` (served `sw` wide) into `dst` (served `vw` wide), each
    /// checked where it is settled.
    fn of(src: &Space, dst: &Space, sw: usize, vw: usize, looked_up: bool) -> Self {
        assert!(
            src.axes().all(|axis| !src.is_dynamic(axis)),
            "Tile::copy_from: a decoding copy reads its window through the values' own axes, which \
             needs their extents at expansion; this window spans a dynamic axis. Copy a region a \
             level cuts to (a stage, a leaf) rather than the whole dynamic tile"
        );
        assert!(
            sw.is_multiple_of(vw),
            "Tile::copy_from: a decoding copy writes each source line as whole destination lines, \
             and a {sw}-wide source line does not split into {vw}-wide ones"
        );
        let kept: Vec<bool> = src.axes().map(|axis| dst.contains(axis)).collect();
        let narrows = src.axes().filter(|&axis| dst.contains(axis)).eq(dst.axes())
            && kept.last() == Some(&true)
            && src.axes().all(|axis| match dst.contains(axis) {
                true => src.extent(axis) == dst.extent(axis),
                false => src.extent(axis) == 1,
            });
        assert!(
            narrows,
            "Tile::copy_from: a decoding copy writes the box it reads, and {src:?} does not \
             narrow to {dst:?} by dropping one-wide axes"
        );
        Decode {
            space: src.clone(),
            kept,
            sw,
            vw,
            looked_up,
        }
    }
}

/// The `c`-th destination line of the source line at `at`, decoded and written: every index
/// replaced by its table entry, then the line scaled at its first value.
#[cube]
fn land<T: Numeric, VW: Size>(
    out: &mut MaskedMut<'_, Vector<T, VW>, CoordsDyn>,
    mem: &Memory<T>,
    at: &Coords<u32>,
    stored: Vector<T, VW>,
    #[comptime] c: usize,
    #[comptime] decode: Decode,
) {
    let rank = comptime!(decode.space.rank());
    let decoded = if comptime!(decode.looked_up) {
        let mut entries = Vector::<T, VW>::empty();
        #[unroll]
        for j in 0..comptime!(decode.vw) {
            let index = u32::cast_from(stored.extract(j));
            entries.insert(j, T::cast_from(mem.codebook.entry(index)));
        }
        entries
    } else {
        stored
    };
    // The line's first value in the source's space, and the same place in the destination's,
    // which lacks the source's one-wide axes.
    let mut first = Coords::<u32>::new();
    let mut write_at = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        let offset = comptime!((if p == rank - 1 { c * decode.vw } else { 0 }) as u32);
        let coord = at.at(p) + offset;
        first.push(coord);
        if comptime!(decode.kept[p]) {
            write_at.push(coord);
        }
    }
    let scale = mem
        .factor
        .at_coords(&first, comptime!(decode.space.clone()));
    out.write(
        as_dyn(&write_at, comptime!(decode.vw)),
        decoded * Vector::<T, VW>::cast_from(scale),
    );
}
