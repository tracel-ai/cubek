use crate::linear_recurrence::{CHANNEL_BOX, LINE_IN_BOX};
use cubecl::tile::Region;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// How channels are cut into boxes: a cube's box holds `units` lines of the `channels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelBox {
    pub units: usize,
    pub channels: usize,
}

/// Which line of the channels a unit of the cube holds: the cube's box starts at the line its
/// index times the units says, and the last box may reach past the channels.
#[derive(CubeType)]
pub(super) struct BoxLines {
    first: usize,
    #[cube(comptime)]
    lines: usize,
}

#[cube]
impl BoxLines {
    pub fn new<N: Size>(cube: &Region, #[comptime] shape: ChannelBox) -> Self {
        let width = N::value();
        BoxLines {
            first: cube.coord(CHANNEL_BOX) * shape.units,
            lines: comptime!(shape.channels / width),
        }
    }

    /// The line `share`'s unit holds, which may lie past the channels.
    pub fn line(&self, share: &Region) -> usize {
        self.first + share.coord(LINE_IN_BOX)
    }

    /// Whether `line` is one of the channels'.
    pub fn holds(&self, line: usize) -> bool {
        line < self.lines
    }
}
