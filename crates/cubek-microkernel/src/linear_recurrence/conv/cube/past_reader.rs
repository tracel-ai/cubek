use crate::linear_recurrence::{POSITION, PROMPT_POSITION, RING_COLUMN, walked};
use cubecl::tile::{Axis, Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Who writes the positions a launch convolves into the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RingWriter {
    /// The conv itself, which writes the newest value of each row as the ring's next column: a
    /// launch of one position a row.
    ConvLaunch,
    /// A launch after the conv, once every position has read the ring's old columns.
    WriteBackLaunch,
}

/// The operands the taps' past is read from, as tiles.
#[derive(CubeType)]
pub struct CausalConvPastTiles {
    /// The row's positions to convolve.
    pub input: Tile,
    /// The ring store, placed at the row's slot.
    pub ring: Tile,
    /// The positions the row has convolved before this launch.
    pub lengths: Tile,
    /// The ring's columns, the one extent of the ring no tile carries.
    pub columns: usize,
}

#[cube]
impl CausalConvPastTiles {
    pub fn at(&self, region: &Region) -> CausalConvPastTiles {
        CausalConvPastTiles {
            input: self.input.at(region),
            ring: self.ring.at(region),
            lengths: self.lengths.at(region),
            columns: self.columns,
        }
    }
}

/// Where a unit reads the values its taps run over: earlier positions of the row's input, then its
/// ring.
#[derive(CubeType)]
pub struct PastReader {
    input: Tile,
    ring: Tile,
    /// The positions the row has convolved before the launch: its ring's fill.
    before: usize,
    /// The cube's place in the row's input.
    position: usize,
    columns: usize,
    #[cube(comptime)]
    writer: RingWriter,
    #[cube(comptime)]
    read_columns: usize,
}

#[cube]
impl PastReader {
    pub fn new(
        cube: &Region,
        tiles: &CausalConvPastTiles,
        #[comptime] writer: RingWriter,
        #[comptime] read_columns: usize,
    ) -> Self {
        PastReader {
            input: tiles.input.clone(),
            ring: tiles.ring.clone(),
            before: walked(&tiles.lengths) as usize,
            position: cube.coord(POSITION),
            columns: tiles.columns,
            writer,
            read_columns,
        }
    }

    /// The `line`-th line at the newest tap: the position's own value.
    pub fn newest<E: Float, N: Size>(&self, line: usize) -> Vector<E, N> {
        read_line::<E, N>(&self.input, PROMPT_POSITION, self.position, line)
    }

    /// The `line`-th line at tap `column`: zero where neither the input nor the row's ring holds it
    /// yet.
    pub fn behind<E: Float, N: Size>(&self, column: usize, line: usize) -> Vector<E, N> {
        let read_columns = self.read_columns;
        let tap = self.position + column;
        let mut behind = Vector::<E, N>::cast_from(0.0f32);
        if tap >= read_columns {
            behind = read_line::<E, N>(&self.input, PROMPT_POSITION, tap - read_columns, line);
        } else if self.before + tap >= read_columns {
            // The column wraps at a runtime length, which no window of the tile places.
            let ring_column = (self.before + tap + self.columns - read_columns) % self.columns;
            behind = read_line::<E, N>(&self.ring, RING_COLUMN, ring_column, line);
        }
        behind
    }

    /// `newest` as the ring's next column where the launch writes the ring.
    pub fn write_newest<E: Float, N: Size>(&self, line: usize, newest: Vector<E, N>) {
        let writer = comptime!(self.writer);
        match writer {
            RingWriter::ConvLaunch => {
                let next = self.before % self.columns;
                let mut column = self.ring.within(RING_COLUMN, next, next + 1);
                column.flat_mut::<E, N>().write(line, newest);
            }
            RingWriter::WriteBackLaunch => {}
        }
    }
}

/// The `line`-th line of the tile at `column` along `axis`.
#[cube]
pub(super) fn read_line<E: Float, N: Size>(
    tile: &Tile,
    #[comptime] axis: Axis,
    column: usize,
    line: usize,
) -> Vector<E, N> {
    let placed = tile.within(axis, column, column + 1);
    placed.flat::<E, N>().read(line)
}
