use crate::linear_recurrence::{CHANNEL_RUN, LINE_IN_RUN};
use cubecl::tile::Region;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};
use cubek_std::Hardware;

/// How channels are cut into runs: a cube's run holds `units` lines of the `channels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelRun {
    pub units: usize,
    pub channels: usize,
}

/// The units a cube deals channel lines between, at most.
const MOST_UNITS: usize = 64;

impl ChannelRun {
    /// The most units one cube of a conv holds on `hw`.
    pub fn most_units(hw: &Hardware) -> usize {
        MOST_UNITS.min(hw.most_units_1d())
    }
}

/// Which line of the channels a unit of the cube holds: the cube's run starts at the line its
/// index times the units says, and the last run may reach past the channels.
#[derive(CubeType)]
pub(super) struct RunLines {
    first: usize,
    #[cube(comptime)]
    lines: usize,
}

#[cube]
impl RunLines {
    pub fn new<N: Size>(cube: &Region, #[comptime] shape: ChannelRun) -> Self {
        let width = N::value();
        RunLines {
            first: cube.coord(CHANNEL_RUN) * shape.units,
            lines: comptime!(shape.channels / width),
        }
    }

    /// The line `share`'s unit holds, which may lie past the channels.
    pub fn line(&self, share: &Region) -> usize {
        self.first + share.coord(LINE_IN_RUN)
    }

    /// Whether `line` is one of the channels'.
    pub fn holds(&self, line: usize) -> bool {
        line < self.lines
    }
}
