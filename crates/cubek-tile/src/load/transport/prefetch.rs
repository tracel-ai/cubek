//! The straight transport split in two, with a unit's registers in between: its lines are read
//! ahead of a contraction ([`fetch_straight`](Memory::fetch_straight)) and written into the stage
//! after it ([`store_fetched`](Memory::store_fetched)).
//!
//! The point is the gap: a stage's global loads are in flight while the contraction before them
//! runs, rather than issued after the barrier that frees their slot.

use cubecl::prelude::*;

use super::cooperative::fill_extent;
use super::padded::{physical_pos, read_stage_line};
use crate::*;

/// Scalars of both operands' stages one unit may hold in registers beside a contraction's own
/// accumulator, summed over the two ([`fetched_scalars`]).
pub const MOST_FETCHED_SCALARS: usize = 64;

/// Scalars one unit of `units` holds in registers for a stage of `elements` read in lines `width`
/// wide, when its fill is fetched ahead of a contraction: the rule a routine asks before choosing
/// that schedule, and the one [`Stages::prefetched`] holds it to.
pub fn fetched_scalars(elements: usize, width: usize, units: usize) -> usize {
    elements.div_ceil(width).div_ceil(units) * width
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// Scalars of this stage one unit holds in registers between [`fetch_straight`] and
    /// [`store_fetched`](Memory::store_fetched): every element of the lines it copies.
    ///
    /// [`fetch_straight`]: Memory::fetch_straight
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetched_scalars(&self) -> comptime_type!(usize) {
        let total_c = self.stage_lines().constant();
        let w = comptime!(self.store.vector_size);
        let units = comptime!(self.access.units);
        comptime!(fetched_scalars(
            fetched_stage_lines(total_c, units) * w,
            w,
            units
        ))
    }

    /// This stage's lines, a count its whole shape folds at expansion.
    fn stage_lines(&self) -> usize {
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        shape
            .product(comptime!((0..plen).collect::<Vec<_>>()))
            .cast::<usize>()
    }

    /// The straight fill's first half: this unit's lines of `src` read into `fetched`, one
    /// `W`-wide line per slot ([`fetched_scalars`](Memory::fetched_scalars) over the width). The
    /// second half, [`store_fetched`](Memory::store_fetched), writes them into this stage.
    ///
    /// Only the copy a matmul stage takes: a plain, direct, unmasked stage that replaces, filled
    /// at its own width from a plain direct source. Anything else is refused at expansion. The
    /// lines are dealt over the spec's `units`, which must be the launch's `CUBE_DIM`
    /// ([`dealt_line`]).
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetch_straight<W: Size>(
        &self,
        src: &Memory<T>,
        #[comptime] space: Space,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        comptime!(assert!(
            self.store.quant.is_none()
                && src.store.quant.is_none()
                && self.store.packing == Packing::Plain
                && src.store.packing == Packing::Plain
                && self.access.whole
                && !self.access.overhang.masks()
                && self.access.write == Write::Replace
                && self.store.vector_size == src.store.vector_size
                && self.projection.is_direct()
                && src.projection.is_direct(),
            "Memory::fetch_straight: only a plain, direct, whole stage filled at its source's \
             width is fetched into registers"
        ));
        let w = comptime!(self.store.vector_size);
        let fetched_width = W::value();
        comptime!(assert_eq!(
            fetched_width, w,
            "Memory::fetch_straight: registers of one width fetch a stage of another"
        ));
        let check = comptime!(src.access.overhang.masks());
        comptime!(fill_extent(&space, w, w, check));
        let shape = self.layout.physical_shape.clone();
        let projection = comptime!(self.layout.projection.clone());
        let units = comptime!(self.access.units);
        let total = self.stage_lines();
        let total_c = total.constant();
        let total_c = comptime!(fetched_stage_lines(total_c, units));
        let s = Masked::new(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        #[unroll]
        for t in 0..comptime!(total_c.div_ceil(units)) {
            let (i, in_stage) = dealt_line(t, units, total, total_c);
            if in_stage {
                fetched[t] = read_stage_line::<T, W, W>(
                    &s,
                    &physical_pos(comptime!(projection.clone()), i, &shape),
                    comptime!(None),
                );
            }
        }
    }

    /// The second half of [`fetch_straight`](Memory::fetch_straight): this unit's lines written
    /// into this stage out of `fetched`. The caller owns the rendezvous, as for a whole fill: a
    /// `sync_cube` must separate the stage's last read from this write, and this write from the
    /// next read.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn store_fetched<W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        let units = comptime!(self.access.units);
        let total = self.stage_lines();
        let total_c = total.constant();
        let total_c = comptime!(fetched_stage_lines(total_c, units));
        let d = self.lines_storage_mut::<T, W>();
        #[unroll]
        for t in 0..comptime!(total_c.div_ceil(units)) {
            let (i, in_stage) = dealt_line(t, units, total, total_c);
            if in_stage {
                d[i] = fetched[t];
            }
        }
    }
}

/// Task `t` of this unit's share of a stage of `total` lines (`total_c` folded) dealt over `units`
/// units, straight-line: the line it moves, and whether that line is in the stage. Only the last
/// task can run past the stage, and only where the units do not divide its lines, so only there
/// is the guard a comparison; elsewhere it is a constant the expansion folds away.
///
/// `units` must be the launch's `CUBE_DIM`, which the kernel cannot check: fewer, and the units
/// past them write past the stage; more, and lines of it are never written.
#[cube]
fn dealt_line(
    #[comptime] t: usize,
    #[comptime] units: usize,
    total: usize,
    #[comptime] total_c: usize,
) -> (usize, bool) {
    let i = UNIT_POS as usize + comptime!(t * units);
    let in_stage = if comptime!((t + 1) * units > total_c) {
        i < total
    } else {
        // A constant, which the `if` it guards folds away.
        true.runtime()
    };
    (i, in_stage)
}

/// The lines of a stage fetched into registers over `units` units: its folded `total`, refused
/// where the shape is not static or the units are not stated.
fn fetched_stage_lines(total: Option<u64>, units: usize) -> usize {
    assert!(
        units > 0,
        "Memory: a stage fetched into registers deals its lines over the launch's units, which \
         this operand's spec does not state: bind it through `Launcher::arg`, or set its `units`"
    );
    total.expect("Memory: a stage fetched into registers has a static shape") as usize
}
