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
/// accumulator, summed over the two ([`UnitLines::scalars`]).
///
/// 128 scalars: as many registers of 32-bit values, half as many of packed 16-bit pairs. That is
/// enough for a GEMM stage 64 deep on eight planes (96 scalars a unit), whose global loads are
/// then in flight across a contraction rather than waited on after it. Whether a stage this large
/// still fits beside its accumulator without spilling is the caller's to measure; a bound only
/// rules out what cannot pay.
pub const MOST_FETCHED_SCALARS: usize = 128;

/// The lines of one stage a single unit moves, when the cube's units take them between them.
///
/// **Unit `u` takes lines `u`, `u + units`, …**, so every unit takes the same count and only the
/// last of them can run past the stage. Holding the two numbers together is what lets the fetch
/// and the store agree on which line a task is without either re-deriving it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UnitLines {
    /// The units the lines are spread over, which must be the launch's `CUBE_DIM`.
    units: usize,
    /// The stage's whole line count.
    lines: usize,
}

impl UnitLines {
    /// The lines of a stage of `lines` spread over `units`.
    ///
    /// # Panics
    ///
    /// Units that were never stated: a stage fetched into registers spreads its lines over the
    /// launch's units, which an operand's spec has to carry.
    pub fn new(lines: usize, units: usize) -> Self {
        assert!(
            units > 0,
            "UnitLines: a stage fetched into registers spreads its lines over the launch's \
             units, which this operand's spec does not state: bind it through `Launcher::arg`, \
             or set its `units`"
        );
        UnitLines { units, lines }
    }

    /// How many lines one unit moves: the tasks a fetch and a store each run.
    pub(crate) fn tasks(self) -> usize {
        self.lines.div_ceil(self.units)
    }

    /// Scalars one unit holds in registers when each line is `width` wide: what a routine asks
    /// before choosing this schedule, and what the stages hold it to.
    pub fn scalars(self, width: usize) -> usize {
        self.tasks() * width
    }

    /// Whether task `t` has to be guarded against running past the stage. Only the last task can,
    /// and only where the units do not divide the lines; elsewhere the guard is a constant the
    /// expansion folds away.
    fn guarded(self, t: usize) -> bool {
        (t + 1) * self.units > self.lines
    }
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// The lines of this stage one unit moves ([`UnitLines`]).
    ///
    /// # Panics
    ///
    /// A stage whose shape does not fold, which has no line count to spread.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn unit_lines(&self) -> comptime_type!(UnitLines) {
        let folded = self.stage_lines().constant();
        let units = comptime!(self.access.units);
        comptime!(UnitLines::new(
            folded.expect("Memory: a stage fetched into registers has a static shape") as usize,
            units,
        ))
    }

    /// Scalars of this stage one unit holds in registers between [`fetch_straight`] and
    /// [`store_fetched`](Memory::store_fetched): every element of the lines it copies.
    ///
    /// [`fetch_straight`]: Memory::fetch_straight
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetched_scalars(&self) -> comptime_type!(usize) {
        let lines = self.unit_lines();
        comptime!(lines.scalars(self.store.vector_size))
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
    /// `W`-wide line per task. The second half, [`store_fetched`](Memory::store_fetched), writes
    /// them into this stage.
    ///
    /// Only the copy a matmul stage takes: a plain, direct, unmasked stage that replaces, filled
    /// at its own width from a plain direct source. Anything else is refused at expansion.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetch_straight<W: Size>(
        &self,
        src: &Memory<T>,
        #[comptime] space: Space,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        self.refuse_unfetchable(src);
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
        let rows = comptime!(self.layout.rows);
        let lines = self.unit_lines();
        let total = self.stage_lines();
        let s = Masked::new(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        #[unroll]
        for t in 0..comptime!(lines.tasks()) {
            let i = task_line(t, comptime!(lines));
            if in_stage(i, total, t, comptime!(lines)) {
                fetched[t] = read_stage_line::<T, W, W>(
                    &s,
                    &physical_pos(comptime!(projection.clone()), rows, i, &shape),
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
        let lines = self.unit_lines();
        let total = self.stage_lines();
        let rows = comptime!(self.layout.rows);
        let shape = self.layout.physical_shape.clone();
        let strides = self.layout.physical_strides.clone();
        let d = self.lines_storage_mut::<T, W>();
        #[unroll]
        for t in 0..comptime!(lines.tasks()) {
            let i = task_line(t, comptime!(lines));
            if in_stage(i, total, t, comptime!(lines)) {
                d[stage_offset(rows, i, &shape, &strides)] = fetched[t];
            }
        }
    }

    /// Refuse a pairing this transport cannot move: everything it rests on, asserted where the
    /// fill would otherwise read or write the wrong cells rather than fail.
    fn refuse_unfetchable(&self, src: &Memory<T>) {
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
    }
}

/// The line task `t` of this unit moves.
#[cube]
fn task_line(#[comptime] t: usize, #[comptime] lines: UnitLines) -> usize {
    UNIT_POS as usize + comptime!(t * lines.units)
}

/// Whether line `i` is one of the stage's `total`. Guarded only where task `t` can run past it
/// ([`UnitLines::guarded`]); elsewhere a constant the `if` folds away.
#[cube]
fn in_stage(i: usize, total: usize, #[comptime] t: usize, #[comptime] lines: UnitLines) -> bool {
    if comptime!(lines.guarded(t)) {
        i < total
    } else {
        true.runtime()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every unit runs the same number of tasks, so the count rounds up: the units that have no
    /// line left still step through the loop, guarded.
    #[test]
    fn every_unit_runs_the_same_tasks() {
        assert_eq!(UnitLines::new(256, 64).tasks(), 4);
        assert_eq!(UnitLines::new(255, 64).tasks(), 4);
        assert_eq!(UnitLines::new(1, 64).tasks(), 1);
    }

    /// The registers a unit holds are its tasks at the line's width.
    #[test]
    fn the_registers_are_the_tasks_at_the_width() {
        assert_eq!(UnitLines::new(256, 64).scalars(4), 16);
        assert_eq!(UnitLines::new(256, 64).scalars(1), 4);
    }

    /// Only the last task can run past the stage, and only where the units do not divide its
    /// lines: everywhere else the guard is a constant.
    #[test]
    fn only_a_last_task_past_the_lines_is_guarded() {
        let exact = UnitLines::new(256, 64);
        for t in 0..exact.tasks() {
            assert!(
                !exact.guarded(t),
                "task {t} of an exact spread needs no guard"
            );
        }
        let ragged = UnitLines::new(200, 64);
        assert_eq!(ragged.tasks(), 4);
        for t in 0..3 {
            assert!(!ragged.guarded(t), "task {t} lies wholly inside the lines");
        }
        assert!(ragged.guarded(3), "the last task runs past 200 lines");
    }

    /// Units nobody stated are refused by name: the spread would otherwise divide by zero.
    #[test]
    #[should_panic(expected = "spreads its lines over the launch's units")]
    fn lines_with_no_units_are_refused() {
        UnitLines::new(256, 0);
    }
}
