//! A clean, launchable [`Tile`] input for tests. You say "a tile over `(M:m,
//! K:k)` with `tile`-sized subtiles" and get a launchable tile: the buffer is a
//! plain `[grid…, tile…]` strided tensor, presented as its logical space via an
//! explicit `tiled_view`. No `Tiler`, no semantic-view juggling.

use cubecl::std::tensor::{
    TensorHandle,
    layout::tiled_view::{TiledViewLaunch, tiled_view},
};
use cubecl::{TestRuntime, client::ComputeClient, zspace::shape};
use cubek_test_utils::TestInput;

use super::tile_dsl::{Axis, Space};

/// A tile-shaped test input: the device buffer plus the logical [`Space`] and
/// sub-tile sizes needed to view it as a tile.
pub struct TileInput {
    handle: TensorHandle<TestRuntime>,
    space: Space,
    subtiles: Vec<u16>,
}

impl TileInput {
    /// A read input over the logical 2-D space `(rows, cols)` with square `tile`
    /// subtiles, filled from `at(i, j)` in semantic coordinates (stored tiled).
    pub fn read(
        client: &ComputeClient<TestRuntime>,
        rows: (Axis, usize),
        cols: (Axis, usize),
        tile: usize,
        at: impl Fn(usize, usize) -> f32,
    ) -> Self {
        let (_, r) = rows;
        let (_, c) = cols;
        let handle = TestInput::builder(client.clone(), shape![r / tile, c / tile, tile, tile])
            .custom(tile_permuted(r, c, tile, at))
            .generate();
        Self::new(handle, rows, cols, tile)
    }

    /// A zero-filled write output over the logical 2-D space `(rows, cols)`.
    pub fn write(
        client: &ComputeClient<TestRuntime>,
        rows: (Axis, usize),
        cols: (Axis, usize),
        tile: usize,
    ) -> Self {
        let (_, r) = rows;
        let (_, c) = cols;
        let handle = TestInput::builder(client.clone(), shape![r / tile, c / tile, tile, tile])
            .zeros()
            .generate_without_host_data();
        Self::new(handle, rows, cols, tile)
    }

    fn new(
        handle: TensorHandle<TestRuntime>,
        rows: (Axis, usize),
        cols: (Axis, usize),
        tile: usize,
    ) -> Self {
        TileInput {
            handle,
            space: Space::new(&[(rows.0, rows.1 as u32), (cols.0, cols.1 as u32)]),
            subtiles: vec![tile as u16, tile as u16],
        }
    }

    /// Launch arg for this tile's view — the buffer seen in its logical space.
    pub fn view(&self) -> TiledViewLaunch<TestRuntime> {
        tiled_view(self.handle.clone().binding(), 0, self.subtiles.clone())
    }

    /// The semantic space the tile lives in.
    pub fn space(&self) -> Space {
        self.space.clone()
    }

    /// The device handle, for reading an output back.
    pub fn handle(&self) -> TensorHandle<TestRuntime> {
        self.handle.clone()
    }
}

/// Lay `at(i, j)` out in `[grid_r, grid_c, tile, tile]` storage order, so a
/// `tiled_view` returns `at(i, j)` at logical `(i, j)`.
pub fn tile_permuted(
    rows: usize,
    cols: usize,
    tile: usize,
    at: impl Fn(usize, usize) -> f32,
) -> Vec<f32> {
    let grid_rows = rows / tile;
    let grid_cols = cols / tile;
    let mut out = vec![0.0; rows * cols];
    for gm in 0..grid_rows {
        for gn in 0..grid_cols {
            for tm in 0..tile {
                for tn in 0..tile {
                    let offset = gm * grid_cols * tile * tile + gn * tile * tile + tm * tile + tn;
                    out[offset] = at(gm * tile + tm, gn * tile + tn);
                }
            }
        }
    }
    out
}
