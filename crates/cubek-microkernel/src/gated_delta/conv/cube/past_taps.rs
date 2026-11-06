use super::tiles::GatedDeltaConvTiles;
use crate::gated_delta::{POSITION, RING_COLUMN, STREAM_POSITION, walked};
use cubecl::tile::{Axis, Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Where a unit reads the values its taps run over: a row's window, whose ring its newest value
/// joins, or the sequence's stream, read by tap and written nowhere.
#[derive(CubeType)]
pub(super) struct PastTaps {
    qkv: Tile,
    window: ComptimeOption<Tile>,
    /// The positions before this one: those the row has convolved, or its place in the stream.
    before: usize,
    columns: usize,
    #[cube(comptime)]
    read_columns: usize,
}

#[cube]
impl PastTaps {
    pub fn new(
        cube: &Region,
        tiles: &GatedDeltaConvTiles,
        #[comptime] read_columns: usize,
    ) -> Self {
        PastTaps {
            qkv: tiles.qkv.clone(),
            window: tiles.window.clone(),
            before: positions_before(cube, &tiles.lengths),
            columns: tiles.columns,
            read_columns,
        }
    }

    /// The `line`-th line at the newest tap: the position's own value.
    pub fn newest<E: Float, N: Size>(&self, line: usize) -> Vector<E, N> {
        #[comptime]
        match &self.window {
            ComptimeOption::Some(_) => self.qkv.flat::<E, N>().read(line),
            ComptimeOption::None => read_line::<E, N>(
                &self.qkv,
                STREAM_POSITION,
                self.before + self.read_columns,
                line,
            ),
        }
    }

    /// The `line`-th line at tap `column`: zero where the row's window does not yet hold it.
    pub fn behind<E: Float, N: Size>(&self, column: usize, line: usize) -> Vector<E, N> {
        let read_columns = self.read_columns;
        #[comptime]
        match &self.window {
            ComptimeOption::Some(window) => {
                // The column wraps at a runtime length, which no window of the tile places.
                let ring_column =
                    (self.before + column + self.columns - read_columns) % self.columns;
                let mut behind = read_line::<E, N>(window, RING_COLUMN, ring_column, line);
                if self.before + column < read_columns {
                    behind = Vector::<E, N>::cast_from(0.0f32);
                }
                behind
            }
            ComptimeOption::None => {
                read_line::<E, N>(&self.qkv, STREAM_POSITION, self.before + column, line)
            }
        }
    }

    /// `newest` as the ring's next column over a window; nothing over a stream.
    pub fn advance<E: Float, N: Size>(&self, line: usize, newest: Vector<E, N>) {
        #[comptime]
        match &self.window {
            ComptimeOption::Some(window) => {
                let next = self.before % self.columns;
                let mut column = window.within(RING_COLUMN, next, next + 1);
                column.flat_mut::<E, N>().write(line, newest);
            }
            ComptimeOption::None => {}
        }
    }
}

/// The `line`-th line of the tile at `column` along `axis`.
#[cube]
fn read_line<E: Float, N: Size>(
    tile: &Tile,
    #[comptime] axis: Axis,
    column: usize,
    line: usize,
) -> Vector<E, N> {
    let placed = tile.within(axis, column, column + 1);
    placed.flat::<E, N>().read(line)
}

/// The positions before the cube's: the row's convolved count over a window, its place in the
/// stream otherwise.
#[cube]
fn positions_before(cube: &Region, lengths: &ComptimeOption<Tile>) -> usize {
    #[comptime]
    match lengths {
        ComptimeOption::Some(lengths) => walked(lengths) as usize,
        ComptimeOption::None => cube.coord(POSITION),
    }
}
