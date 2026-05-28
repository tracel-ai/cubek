//! A clean, launchable [`Tile`] input for tests. Describe a tile as "a [`Space`]
//! with `tile_edge`-sized square subtiles" and get a launchable tile: the buffer
//! is a plain `[grid…, tile…]` strided tensor, presented in its logical space via
//! an explicit `tiled_view`. No `Tiler`, no semantic-view juggling.

use cubecl::std::tensor::{
    TensorHandle,
    layout::tiled_view::{TileSpec, TiledViewLaunch, TiledViewLayout},
};
use cubecl::{TestRuntime, client::ComputeClient, zspace::shape};
use cubek_test_utils::TestInput;

use super::tile_dsl::Space;

/// A tile-shaped test input: the device buffer plus the logical [`Space`] it's
/// viewed in. The sub-tile sizes live in the buffer's trailing dims, so the view
/// reads them from there.
pub struct TileInput {
    handle: TensorHandle<TestRuntime>,
    space: Space,
}

impl TileInput {
    /// Start building a tile over `space`. Set the subtile size with
    /// [`tile_edge`](TileInputBuilder::tile_edge), then a data finalizer
    /// ([`arange`](TileInputBuilder::arange) / [`zeros`](TileInputBuilder::zeros)).
    pub fn builder(client: &ComputeClient<TestRuntime>, space: Space) -> TileInputBuilder {
        TileInputBuilder {
            client: client.clone(),
            space,
            tile_edge: None,
        }
    }

    /// Launch arg for this tile's view — the buffer seen in its logical space.
    /// Every logical axis is tiled, so `num_tiled = space.rank()`.
    pub fn view(&self) -> TiledViewLaunch<TestRuntime> {
        TiledViewLaunch::new_tensor::<TiledViewLayout>(
            self.handle.clone().binding().into_tensor_arg(),
            TileSpec {
                start_axis: 0,
                num_tiled: self.space.rank(),
            },
        )
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

/// Fluent builder for a [`TileInput`]: a [`Space`], a square subtile edge, and a
/// data finalizer that fills the `[grid…, tile…]` buffer.
pub struct TileInputBuilder {
    client: ComputeClient<TestRuntime>,
    space: Space,
    tile_edge: Option<usize>,
}

impl TileInputBuilder {
    /// Square subtiles of edge `tile_edge` on every axis: the buffer becomes
    /// `[grid…, tile…]` with `grid = extent / tile_edge` per axis.
    pub fn tile_edge(mut self, tile_edge: usize) -> Self {
        self.tile_edge = Some(tile_edge);
        self
    }

    /// Semantic row-major `0, 1, 2, …` over the space, stored tiled so a
    /// `tiled_view` reads `value(i, j) = i * cols + j` back at logical `(i, j)`.
    pub fn arange(self) -> TileInput {
        let (rows, cols) = self.extents();
        let tile = self
            .tile_edge
            .expect("TileInput: call .tile_edge(...) before a data finalizer");
        let semantic: Vec<f32> = (0..(rows * cols)).map(|x| x as f32).collect();
        let tiled = tile_permute(&semantic, rows, cols, tile);
        self.finalize(tiled)
    }

    /// All-zeros tiled buffer — e.g. a matmul output.
    pub fn zeros(self) -> TileInput {
        let (rows, cols) = self.extents();
        self.finalize(vec![0.0; rows * cols])
    }

    fn extents(&self) -> (usize, usize) {
        assert_eq!(self.space.rank(), 2, "TileInput supports 2-D spaces only");
        let rows = self.space.extent(self.space.axis_at(0));
        let cols = self.space.extent(self.space.axis_at(1));
        (rows, cols)
    }

    fn finalize(self, tiled: Vec<f32>) -> TileInput {
        let (rows, cols) = self.extents();
        let tile = self
            .tile_edge
            .expect("TileInput: call .tile_edge(...) before a data finalizer");
        let handle = TestInput::builder(
            self.client.clone(),
            shape![rows / tile, cols / tile, tile, tile],
        )
        .custom(tiled)
        .generate_without_host_data();
        TileInput {
            handle,
            space: self.space,
        }
    }
}

/// Rearrange a row-major `rows × cols` buffer into `[grid_r, grid_c, tile, tile]`
/// storage order, so a `tiled_view` reads `semantic[i * cols + j]` back at logical
/// `(i, j)`. Used for tile inputs and for the expected (matmul) result.
pub fn tile_permute(semantic: &[f32], rows: usize, cols: usize, tile: usize) -> Vec<f32> {
    let grid_rows = rows / tile;
    let grid_cols = cols / tile;
    let mut out = vec![0.0; rows * cols];
    for gm in 0..grid_rows {
        for gn in 0..grid_cols {
            for tm in 0..tile {
                for tn in 0..tile {
                    let offset = gm * grid_cols * tile * tile + gn * tile * tile + tm * tile + tn;
                    let (i, j) = (gm * tile + tm, gn * tile + tn);
                    out[offset] = semantic[i * cols + j];
                }
            }
        }
    }
    out
}
