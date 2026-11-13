use super::past_reader::{CausalConvPastTiles, read_line};
use super::run_lines::{ChannelRun, RunLines};
use crate::linear_recurrence::{PROMPT_POSITION, RING_COLUMN, RING_POSITION, walked};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// A row's last `taps - 1` valid positions of a prompt, copied into its ring once every position
/// has read the ring's old columns, so the copy is its own launch.
#[derive(CubeType)]
pub struct RingWriteBack {
    input: Tile,
    ring: Tile,
    before: usize,
    /// The positions of the prompt that are real, the rest right padding, at most the prompt's.
    valid: usize,
    /// Which of the last `taps - 1` valid positions the cube copies, oldest first.
    written: usize,
    columns: usize,
    #[cube(comptime)]
    read_columns: usize,
}

#[cube]
impl RingWriteBack {
    pub fn new(
        cube: &Region,
        tiles: &CausalConvPastTiles,
        valid: &Tile,
        positions: usize,
        #[comptime] read_columns: usize,
    ) -> Self {
        RingWriteBack {
            input: tiles.input.clone(),
            ring: tiles.ring.clone(),
            before: walked(&tiles.lengths) as usize,
            valid: usize::min(walked(valid) as usize, positions),
            written: cube.coord(RING_POSITION),
            columns: tiles.columns,
            read_columns,
        }
    }

    /// The `line`-th line of the cube's position as the ring's next column, where the row has
    /// that many valid positions.
    pub fn write<E: Float, N: Size>(&self, line: usize) {
        let read_columns = self.read_columns;
        if self.valid + self.written >= read_columns {
            let position = self.valid + self.written - read_columns;
            let value = read_line::<E, N>(&self.input, PROMPT_POSITION, position, line);
            let next = (self.before + position) % self.columns;
            let mut column = self.ring.within(RING_COLUMN, next, next + 1);
            column.flat_mut::<E, N>().write(line, value);
        }
    }
}

/// One cube's run of channel lines of one of a prompt's last `taps - 1` positions, copied into the
/// row's ring.
#[cube]
pub fn cube_ring_write_back_microkernel<E: Float, N: Size>(
    cube: &Region,
    tiles: &CausalConvPastTiles,
    valid: &Tile,
    positions: usize,
    #[comptime] channel_run: ChannelRun,
    #[comptime] read_columns: usize,
) {
    let write_back = RingWriteBack::new(cube, tiles, valid, positions, read_columns);
    let run_lines = RunLines::new::<N>(cube, channel_run);
    for share in cube {
        let line = run_lines.line(&share);
        if run_lines.holds(line) {
            write_back.write::<E, N>(line);
        }
    }
}
