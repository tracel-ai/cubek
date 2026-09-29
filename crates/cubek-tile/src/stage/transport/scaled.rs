//! The decoding copy: a source carrying a table ([`Tile::lookup`]) or scales ([`Tile::mul`])
//! copied into a destination that holds plain values.
//!
//! This is where a kernel decodes on purpose: `stage.copy_from(&w.lookup(&t).mul(&scales))` states
//! the decode at the copy, and nothing decodes behind a read the kernel did not write.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// [`copy_from`](Tile::copy_from) a source carrying a table or scales: each source load is
    /// unpacked, and every destination line it holds replaces its values by their table entries,
    /// multiplies them by the scale at its first value, and lands.
    ///
    /// A load is the source's [`vector_tile`](Tile::vector_tile): a run along the innermost axis,
    /// or a rectangle of the tiles its buffer is stored in (an NVFP4 load of 16 values along `K`
    /// by 2 columns), whose values land a destination line at a time, each where it belongs.
    ///
    /// **The two span one box.** The source may carry axes the destination does not, each one
    /// wide (a batch or head the cube already fixed).
    pub(crate) fn copy_scaled_from(&mut self, src: &Tile<T>) {
        let load = src.vector_tile();
        let sw = comptime!(load.values());
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

    /// The copy, over `stored`: the source's loads as its buffer holds them.
    fn copy_decoded<I: Numeric, WP: Size>(
        &mut self,
        src: &Tile<T>,
        stored: Masked<'_, Vector<I, WP>, CoordsDyn>,
    ) {
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        let (sw, vw) = comptime!((load.values(), lands.values()));
        let size!(VW) = vw;
        let packing = src.packing();
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let rank = comptime!(space.rank());
        let mem = src.mem("Tile::copy_from");
        let along = comptime!(dst.axis_at(dst.rank() - 1));
        let varies = mem.factor.varies_along(comptime!(along));
        comptime!(check_decoding_copy(&space, &dst, &load, vw, varies));
        let mut out = self.nd_mut::<VW>();
        for line in range_stepped(UNIT_POS, load.count(&space), CUBE_DIM) {
            let start = load.start(line, &space);
            let held = stored.read(load.index(&start, &space));
            // The destination lines this load holds, each at its first value, `offset` its
            // position in the load; and where it lands, the same place without the source's
            // one-wide axes.
            #[unroll]
            for c in 0..comptime!(sw / vw) {
                let offset = comptime!(c * vw);
                let mut at = Coords::<u32>::new();
                let mut to = Coords::<u32>::new();
                #[unroll]
                for p in 0..rank {
                    let inner = comptime!(load.offset_along(offset, space.axis_at(p)) as u32);
                    let coord = start.at(p) + inner;
                    at.push(coord);
                    if comptime!(dst.contains(space.axis_at(p))) {
                        to.push(coord);
                    }
                }
                let values = values_at::<T, I, WP, VW>(held, offset, packing);
                let scale = mem.factor.at_coords(&at, comptime!(space.clone()));
                let decoded = mem.codebook.entries(values) * Vector::<T, VW>::cast_from(scale);
                out.write(lands.index(&to, &dst), decoded);
            }
        }
    }
}

/// What a decoding copy from `src`, read in `load`s, into `dst` (written `vw` wide) rests on.
/// `varies` is whether the scales vary along the destination's innermost axis.
fn check_decoding_copy(src: &Space, dst: &Space, load: &VectorTile, vw: usize, varies: bool) {
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
    // A destination line is `vw` consecutive values along its innermost axis, so it sits inside
    // one load only where that is the load's finest axis and its extent there holds whole lines.
    let along = dst.axis_at(dst.rank() - 1);
    let &(finest, run) = load
        .extents()
        .first()
        .expect("a load holds at least one value");
    assert!(
        vw == 1 || (finest == along && run.is_multiple_of(vw)),
        "Tile::copy_from: a decoding copy reads each destination line out of one load, and a \
         {vw}-wide line along {along:?} does not sit inside a load of {:?}",
        load.extents()
    );
    // One scale a destination line, at its first value.
    assert!(
        vw == 1 || !varies,
        "Tile::copy_from: a destination line runs {vw} values along {along:?}, which the scales \
         vary along, so one line lies under several scales; write the destination one value a \
         line, or omit {along:?} from the scales"
    );
}
