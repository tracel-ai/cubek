//! The decoding copy through a table or scales ([`Tile::lookup`], [`Tile::mul`]).

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// [`copy_from`](Tile::copy_from) a source carrying a table or scales.
    /// The source may carry extra axes only if each is one wide.
    pub(crate) fn copy_scaled_from(&mut self, src: &Tile<T>) {
        let (sw, vw) = (src.vector_size(), self.vector_size());
        comptime!(check_decoding_copy(
            &src.place.space,
            &self.place.space,
            sw,
            vw
        ));
        let packing = src.packing();
        match comptime!(packing) {
            Packing::Plain => {
                let size!(SW) = sw;
                self.copy_decoded(src, src.nd_packed::<SW>(comptime!(Guard::Checked)));
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = comptime!(packing.physical(sw));
                self.copy_decoded(src, src.nd_words::<WP>(comptime!(Guard::Checked)));
            }
        }
    }

    /// The copy, over `stored`: the source's lines as its buffer holds them.
    fn copy_decoded<I: Numeric, WP: Size>(
        &mut self,
        src: &Tile<T>,
        stored: Masked<'_, Vector<I, WP>, CoordsDyn>,
    ) {
        let (sw, vw) = (src.vector_size(), self.vector_size());
        let size!(VW) = vw;
        let packing = src.packing();
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let rank = comptime!(space.rank());
        let mem = src.mem("Tile::copy_from");
        let mut out = self.nd_mut::<VW>();
        let lines = comptime!(line_extents(&space, sw, 0, rank));
        let count = comptime!(lines.iter().product::<usize>() as u32);
        for line in range_stepped(UNIT_POS, count, CUBE_DIM) {
            let start = coords_of_line(line, comptime!(lines.clone()), sw);
            let held = stored.read(as_dyn(&start, sw));
            // The destination lines this source line holds, and where they land.
            #[unroll]
            for c in 0..comptime!(sw / vw) {
                let offset = comptime!(c * vw);
                let mut at = Coords::<u32>::new();
                let mut to = Coords::<u32>::new();
                #[unroll]
                for p in 0..rank {
                    let inner = comptime!((if p == rank - 1 { offset } else { 0 }) as u32);
                    let coord = start.at(p) + inner;
                    at.push(coord);
                    if comptime!(dst.contains(space.axis_at(p))) {
                        to.push(coord);
                    }
                }
                let values = values_at::<T, I, WP, VW>(held, offset, packing);
                let scale = mem.factor.at_coords(&at, comptime!(space.clone()));
                let decoded = mem.codebook.entries(values) * Vector::<T, VW>::cast_from(scale);
                out.write(as_dyn(&to, vw), decoded);
            }
        }
    }
}

/// What a decoding copy from `src` (served `sw` wide) into `dst` (served `vw` wide) rests on.
fn check_decoding_copy(src: &Space, dst: &Space, sw: usize, vw: usize) {
    assert!(
        src.axes().all(|axis| !src.is_dynamic(axis)),
        "Tile::copy_from: a decoding copy reads its window through the values' own axes, which \
         needs their extents at expansion; this window spans a dynamic axis. Copy a region a level \
         cuts to (a stage, a leaf) rather than the whole dynamic tile"
    );
    assert!(
        src.narrows_to(dst),
        "Tile::copy_from: a decoding copy writes the box it reads, and {src:?} does not narrow to \
         {dst:?} by dropping one-wide axes"
    );
    assert!(
        sw.is_multiple_of(vw),
        "Tile::copy_from: a decoding copy reads each destination line out of one source line, and \
         a {vw}-wide line does not sit inside {sw}-wide ones"
    );
}
