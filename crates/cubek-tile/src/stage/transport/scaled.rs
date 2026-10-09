//! The decoding copy through a table or scales ([`Tile::lookup`], [`Tile::mul`]).

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// [`copy_from`](Tile::copy_from) a source carrying a table or scales, read in the source's
    /// own loads ([`vector_tile`](Tile::vector_tile)), a run or a rectangle of its stored tiles.
    /// The source may carry extra axes only if each is one wide.
    pub(crate) fn copy_scaled_from(&mut self, src: &Tile<T>) {
        // It decodes in the unit that loads each line; an async copy lands the bytes as they lie.
        let delivery = src.delivery();
        comptime!(assert!(
            !delivery.is_async(),
            "Tile::copy_from: a decoding copy cannot be delivered {delivery:?}; deliver it \
             SyncPerUnit, or stage it as it lies and decode out of the stage"
        ));
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

    /// The copy, over `stored`: the source's loads as its buffer holds them, cyclically across
    /// the units that fill this tile.
    fn copy_decoded<I: Numeric, WP: Size>(
        &mut self,
        src: &Tile<T>,
        stored: Masked<'_, Vector<I, WP>, CoordsDyn>,
    ) {
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        let vw = comptime!(lands.values());
        let size!(VW) = vw;
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let along = comptime!(dst.axis_at(dst.rank() - 1));
        let varies = src
            .mem("Tile::copy_from")
            .factor
            .varies_along(comptime!(along));
        comptime!(check_decoding_copy(&space, &dst, &load, vw, varies));
        let fill = self.fill_units();
        let units = comptime!(fill.count);
        let loads = comptime!(load.counts(&space).iter().product::<usize>());
        let mut out = self.nd_mut::<VW>();
        // A comptime worker count unrolls the loads, as a straight fill unrolls its lines: a
        // runtime stride stalls each store on Metal's in-order pipe.
        if comptime!(units > 0 && loads.div_ceil(units) <= 8) {
            #[unroll]
            for t in 0..comptime!(loads.div_ceil(units)) {
                let line = FillUnits::worker(fill) as u32 + comptime!((t * units) as u32);
                if comptime!((t + 1) * units <= loads) {
                    src.decode_load_into::<I, WP, VW>(
                        &stored,
                        &mut out,
                        line,
                        comptime!(dst.clone()),
                        comptime!(lands.clone()),
                    );
                } else {
                    if line < comptime!(loads as u32) {
                        src.decode_load_into::<I, WP, VW>(
                            &stored,
                            &mut out,
                            line,
                            comptime!(dst.clone()),
                            comptime!(lands.clone()),
                        );
                    }
                }
            }
        } else {
            let first = FillUnits::worker(fill) as u32;
            let stride = FillUnits::workers(fill) as u32;
            for line in range_stepped(first, comptime!(loads as u32), stride) {
                src.decode_load_into::<I, WP, VW>(
                    &stored,
                    &mut out,
                    line,
                    comptime!(dst.clone()),
                    comptime!(lands.clone()),
                );
            }
        }
    }

    /// This source's `line`-th load, read out of `stored`, decoded under its table and scales
    /// into `out`, a window spanning `dst` written in `lands`.
    fn decode_load_into<I: Numeric, WP: Size, VW: Size>(
        &self,
        stored: &Masked<'_, Vector<I, WP>, CoordsDyn>,
        out: &mut MaskedMut<'_, Vector<T, VW>, CoordsDyn>,
        line: u32,
        #[comptime] dst: Space,
        #[comptime] lands: VectorTile,
    ) {
        let load = self.vector_tile();
        let (sw, vw) = comptime!((load.values(), lands.values()));
        // Read placed: an `e2m1` value decoded without its lift, which the scale carries, one
        // multiply a line rather than one a value.
        let stored_as = self.packing();
        let packing = comptime!(stored_as.placed());
        let lift = comptime!(packing.lift());
        let space = comptime!(self.place.space.clone());
        let rank = comptime!(space.rank());
        let mem = self.mem("Tile::copy_from");
        let start = load.start(line, &space);
        let held = stored.read(load.index(&start, &space));
        // The destination lines this load holds, `offset` their place in it, and where they land.
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
            let scale = mem.factor.at_coords(&at, comptime!(space.clone())) * lift;
            // Multiplied in `f32`, rounded to `T` once: a placed `e2m1` value is a lift short,
            // which the scale carries, and the lift times a scale is past a half's range where
            // its product with the value is not.
            let scaled = Vector::<f32, VW>::cast_from(mem.codebook.entries(values))
                * Vector::<f32, VW>::cast_from(scale);
            let decoded = Vector::<T, VW>::cast_from(scaled);
            out.write(lands.index(&to, &dst), decoded);
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
