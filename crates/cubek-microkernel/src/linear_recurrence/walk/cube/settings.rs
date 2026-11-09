use core::num::NonZeroUsize;

/// The most vector lines of a row or a state a unit holds in registers: past it the registers, not
/// the data, bound the kernel, and its unrolled walk takes longer to compile than the launch it
/// serves takes to run.
pub const MAX_HELD_LINES_PER_UNIT: usize = 16;

/// How much of a run's state one unit holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeWalkMicrokernelSettings {
    /// A target, not a demand: the `f32` cells of the state one unit holds in registers. Fewer
    /// cells is more units a run, and wider meetings.
    pub cells: NonZeroUsize,
}

impl CubeWalkMicrokernelSettings {
    /// 64 cells a unit: a 128-row head read four columns wide in runs of eight units.
    pub const DEFAULT: Self = Self::new(64);
    /// The fewest cells whose lines stay within [`MAX_HELD_LINES_PER_UNIT`] at any width.
    pub const FLOOR: Self = Self::new(MAX_HELD_LINES_PER_UNIT);

    pub const fn new(cells: usize) -> Self {
        Self {
            cells: NonZeroUsize::new(cells).unwrap(),
        }
    }

    /// These settings with no more than `cells` cells a unit.
    pub fn capped_at(self, cells: usize) -> Self {
        Self::new(self.cells.get().min(cells))
    }
}

impl core::fmt::Display for CubeWalkMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cells={}", self.cells)
    }
}
