use crate::linear_recurrence::{POSITION, RING_COLUMN, STREAM_POSITION, walked};
use cubecl::tile::{Axis, Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// What the taps' past is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PastKind {
    /// One position a row: the taps behind it are the row's ring, and its own value joins the ring.
    Window,
    /// A run of positions a row: the taps behind a position are earlier positions of the run,
    /// then the row's ring. The conv only reads the ring; a write back fills it.
    Prompt,
}

/// The operands the taps' past is read from, as tiles.
#[derive(CubeType)]
pub struct PastTiles {
    /// The row's newest values over a window, its run of positions over a prompt.
    pub input: Tile,
    /// The ring store, placed at the row's slot.
    pub ring: Tile,
    /// The positions the row has convolved before its window or prompt.
    pub lengths: Tile,
    /// The ring's columns, the one extent of the ring no tile carries.
    pub columns: usize,
}

#[cube]
impl PastTiles {
    pub fn at(&self, region: &Region) -> PastTiles {
        PastTiles {
            input: self.input.at(region),
            ring: self.ring.at(region),
            lengths: self.lengths.at(region),
            columns: self.columns,
        }
    }
}

/// Where a unit reads the values its taps run over, and over a window where its newest value
/// joins the ring.
#[derive(CubeType)]
pub struct PastTaps {
    input: Tile,
    ring: Tile,
    /// The positions the row has convolved before the cube's: its ring's fill.
    before: usize,
    /// The cube's place in its prompt, zero over a window.
    position: usize,
    columns: usize,
    #[cube(comptime)]
    kind: PastKind,
    #[cube(comptime)]
    read_columns: usize,
}

#[cube]
impl PastTaps {
    pub fn new(
        cube: &Region,
        tiles: &PastTiles,
        #[comptime] kind: PastKind,
        #[comptime] read_columns: usize,
    ) -> Self {
        let position = match kind {
            PastKind::Window => 0usize,
            PastKind::Prompt => cube.coord(POSITION),
        };
        PastTaps {
            input: tiles.input.clone(),
            ring: tiles.ring.clone(),
            before: walked(&tiles.lengths) as usize,
            position,
            columns: tiles.columns,
            kind,
            read_columns,
        }
    }

    /// The `line`-th line at the newest tap: the position's own value.
    pub fn newest<E: Float, N: Size>(&self, line: usize) -> Vector<E, N> {
        let kind = comptime!(self.kind);
        match kind {
            PastKind::Window => self.input.flat::<E, N>().read(line),
            PastKind::Prompt => {
                read_line::<E, N>(&self.input, STREAM_POSITION, self.position, line)
            }
        }
    }

    /// The `line`-th line at tap `column`: zero where neither the prompt nor the row's ring holds
    /// it yet.
    pub fn behind<E: Float, N: Size>(&self, column: usize, line: usize) -> Vector<E, N> {
        let read_columns = self.read_columns;
        let kind = comptime!(self.kind);
        match kind {
            PastKind::Window => {
                // The column wraps at a runtime length, which no window of the tile places.
                let ring_column =
                    (self.before + column + self.columns - read_columns) % self.columns;
                let mut behind = read_line::<E, N>(&self.ring, RING_COLUMN, ring_column, line);
                if self.before + column < read_columns {
                    behind = Vector::<E, N>::cast_from(0.0f32);
                }
                behind
            }
            PastKind::Prompt => {
                let tap = self.position + column;
                let mut behind = Vector::<E, N>::cast_from(0.0f32);
                if tap >= read_columns {
                    behind =
                        read_line::<E, N>(&self.input, STREAM_POSITION, tap - read_columns, line);
                } else if self.before + tap >= read_columns {
                    let ring_column =
                        (self.before + tap + self.columns - read_columns) % self.columns;
                    behind = read_line::<E, N>(&self.ring, RING_COLUMN, ring_column, line);
                }
                behind
            }
        }
    }

    /// `newest` as the ring's next column over a window; nothing over a prompt, whose ring a later
    /// write back fills.
    pub fn advance<E: Float, N: Size>(&self, line: usize, newest: Vector<E, N>) {
        let kind = comptime!(self.kind);
        match kind {
            PastKind::Window => {
                let next = self.before % self.columns;
                let mut column = self.ring.within(RING_COLUMN, next, next + 1);
                column.flat_mut::<E, N>().write(line, newest);
            }
            PastKind::Prompt => {}
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
