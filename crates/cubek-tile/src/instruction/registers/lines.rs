//! What the register block reads: one line of an operand, at a position.
//!
//! A plain operand hands its line back. A scaled one multiplies its scales in first, and does it
//! here rather than under a view because a scale line covers several value lines: which lane of it
//! a line takes has to be a constant, and only a caller walking a comptime run knows that. `run`
//! is that position, and an operand with nothing to index by it ignores it.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

/// One operand's lines as the contraction reads them.
#[cube]
pub(crate) trait Lines<E: Numeric, V: Size>: CubeType {
    /// The line at `pos`, folded with whatever this operand carries beside its values. `run` names
    /// this line's ordinal along the edge the fold is indexed by, so the lane it selects is a
    /// constant.
    fn line(&self, pos: Coords2d, #[comptime] run: usize) -> Vector<E, V>;

    /// How many folds one read of them serves. Past one, `run` has to be this line's real ordinal
    /// rather than anything, so the caller walks its lines under a constant.
    fn lanes(&self) -> comptime_type!(usize);
}

#[cube]
impl<'a, E: Numeric, V: Size> Lines<E, V> for MaskedView<'a, Vector<E, V>, Coords2d> {
    fn line(&self, pos: Coords2d, #[comptime] _run: usize) -> Vector<E, V> {
        self.read(pos)
    }

    fn lanes(&self) -> comptime_type!(usize) {
        1usize
    }
}

/// An operand read against its scales: the values' line, times the scale covering it.
///
/// The scales are read through their own matrix, whose columns count blocks where the values'
/// count lines. Which *lane* of a scale line a value line takes is a constant the caller knows and
/// the view could not: `run` is the value line's ordinal along the shared edge, and the lane falls
/// out of it. Which scale *line* to read is only an address, so it stays runtime.
#[derive(CubeType)]
pub(crate) struct ScaledLines<'a, E: Numeric, V: Size, S: Numeric, SW: Size> {
    values: MaskedView<'a, Vector<E, V>, Coords2d>,
    scales: MaskedView<'a, Vector<S, SW>, Coords2d>,
    /// Value lines one scale covers along the shared edge.
    #[cube(comptime)]
    lines_per_scale: usize,
    /// Scales one line of them serves, which is the width the scales are read at.
    #[cube(comptime)]
    lanes: usize,
}

#[cube]
impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> ScaledLines<'a, E, V, S, SW> {
    pub fn new(
        values: MaskedView<'a, Vector<E, V>, Coords2d>,
        scales: MaskedView<'a, Vector<S, SW>, Coords2d>,
        #[comptime] lines_per_scale: usize,
        #[comptime] lanes: usize,
    ) -> Self {
        comptime!(assert!(
            lines_per_scale > 0,
            "ScaledLines: one scale covers less than a whole line of values, so a line straddles \
             two scales"
        ));
        ScaledLines::<'a, E, V, S, SW> {
            values,
            scales,
            lines_per_scale,
            lanes,
        }
    }
}

#[cube]
impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> Lines<E, V> for ScaledLines<'a, E, V, S, SW> {
    fn line(&self, pos: Coords2d, #[comptime] run: usize) -> Vector<E, V> {
        let value = self.values.read(pos);
        let (row, col) = pos;
        let per_line = comptime!(self.lines_per_scale * self.lanes);
        let scale = self.scales.read((row, col / comptime!(per_line as u32)));
        let lane = comptime!((run / self.lines_per_scale) % self.lanes);
        value * Vector::<E, V>::cast_from(scale.extract(lane))
    }

    fn lanes(&self) -> comptime_type!(usize) {
        self.lanes
    }
}
