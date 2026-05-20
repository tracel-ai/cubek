use cubecl::{
    self,
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand, chain::Chain},
    zspace::metadata::Metadata,
};
use cubecl::{TestRuntime, prelude::*};
use cubecl::{
    std::tensor::{AsView, AsViewExpand, AsViewMut, AsViewMutExpand},
    zspace::shape,
};
use cubek_test_utils::{
    HostData, HostDataType, LayoutSpec, TestInput, TileSpec, assert_equals_approx,
};

#[test]
fn read_rowmajor_tensor_as_tiled_layout() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let matrix_len = 4;
    let shape = shape![matrix_len, matrix_len];

    // Input stays row-major: arange writes a contiguous 0..16 into memory.
    let input_handle = TestInput::builder(client.clone(), shape.clone())
        .arange()
        .generate();

    let dtype = f32::as_type_native_unchecked().storage_type();

    // Output's layout includes the tile spec — the resulting metadata is the
    // rank-expanded tiled layout, produced at build time.
    let tile = TileSpec::new(0, vec![2u16, 2]);
    let output_handle = TestInput::builder(client.clone(), shape.clone())
        .layout(LayoutSpec::tiled(LayoutSpec::RowMajor, tile.clone()))
        .zeros()
        .generate_without_host_data();

    let cube_count = CubeCount::new_single();
    let cube_dim = CubeDim::new_single();
    let vector_size = 1;

    launch_read_tensor_as_tiled::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        output_handle.metadata.as_ref().clone(),
        matrix_len,
        dtype,
        vector_size,
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

    #[rustfmt::skip]
    let expected_values = [
        0.0, 1.0, 4.0, 5.0,
        2.0, 3.0, 6.0, 7.0,
        8.0, 9.0, 12.0, 13.0,
        10.0, 11.0, 14.0, 15.0,
    ].to_vec();

    // The expected buffer is identical to the output buffer, just laid out
    // visually as the tiled-physical 4x4 above. Tile it so the metadata's
    // rank-expanded shape matches the output for the comparison.
    let (_, expected_values) = TestInput::builder(client, shape)
        .layout(LayoutSpec::tiled(LayoutSpec::RowMajor, tile))
        .custom(expected_values)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected_values, 1e-6)
        .as_test_outcome()
        .enforce()
}

#[derive(CubeType, Clone, Copy)]
pub struct RowMajorLayout {
    width: usize,
    height: usize,
    vector_size: usize,
}

#[cube]
impl RowMajorLayout {
    pub fn new(width: usize, height: usize, vector_size: usize) -> Self {
        RowMajorLayout {
            width,
            height,
            vector_size,
        }
    }
}

#[cube]
impl Layout for RowMajorLayout {
    type Coordinates = (usize, usize);

    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        (self.width * pos.0 + pos.1) / self.vector_size
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let is_valid = pos.0 < self.height && pos.1 < self.width;
        (self.to_source_pos(pos), is_valid)
    }

    fn shape(&self) -> Self::Coordinates {
        (self.width, self.height)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }
}

#[cube(launch)]
fn launch_read_tensor_as_tiled<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: &mut Tensor<Vector<N, S>>,
    #[comptime] metadata: Metadata,
    #[comptime] matrix_len: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] vector_size: usize,
) {
    let tiler = metadata.tiler.clone().unwrap();

    let mut physical_shape = CoordsDyn::new();
    #[unroll]
    for i in 0..metadata.shape.rank() {
        physical_shape.push(comptime!(metadata.shape[i] as u32));
    }

    let mut physical_strides = CoordsDyn::new();
    #[unroll]
    for i in 0..metadata.strides.rank() {
        physical_strides.push(comptime!(metadata.strides[i] as u32));
    }

    let mut tiles = CoordsDyn::new();
    #[unroll]
    for i in 0..tiler.tile_size.len() {
        tiles.push(comptime!(tiler.tile_size[i] as u32));
    }

    // Semantic (rank R) -> physical (rank R + n) -> 1D buffer.
    let semantic_to_physical =
        TiledLayout::new(physical_shape.clone(), tiler.start_axis as usize, tiles);
    let physical_to_buffer = StridedNdLayout::new(physical_shape, physical_strides);
    let tiled_layout =
        Chain::<StridedNdLayout, TiledLayout>::new(physical_to_buffer, semantic_to_physical);

    let row_major = RowMajorLayout::new(matrix_len, matrix_len, vector_size);

    let input_view = input.view(row_major);
    let output_view = output.view_mut(tiled_layout);

    #[unroll]
    for i in 0..matrix_len {
        #[unroll]
        for j in 0..matrix_len {
            let mut coords = CoordsDyn::new();
            coords.push(i as u32);
            coords.push(j as u32);
            let value = input_view.read((i.runtime(), j.runtime()));
            output_view.write(coords, value);
        }
    }
}

/// Maps semantic coordinates (rank R) to physical coordinates (rank R + n).
///
/// For each axis in `[start_axis, start_axis + n)`, the semantic coordinate is
/// split into a grid component and a tile component: `c -> (c / T, c % T)`.
/// The physical layout is `[Pre-axes, Grid-axes, Tile-axes, Post-axes]`.
#[derive(CubeType, Clone)]
pub struct TiledLayout {
    physical_shape: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    tiles: CoordsDyn,
}

#[cube]
impl TiledLayout {
    pub fn new(
        physical_shape: CoordsDyn,
        #[comptime] start_axis: usize,
        tiles: CoordsDyn,
    ) -> TiledLayout {
        TiledLayout {
            physical_shape,
            start_axis,
            tiles,
        }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = CoordsDyn;

    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        #[comptime]
        let n = self.tiles.len();
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut physical = CoordsDyn::new();

        #[unroll]
        for i in 0..self.start_axis {
            physical.push(pos[i]);
        }

        #[unroll]
        for i in 0..n {
            let tile_size = self.tiles[i];
            physical.push(pos[comptime!(self.start_axis + i)] / tile_size);
        }

        #[unroll]
        for i in 0..n {
            let tile_size = self.tiles[i];
            physical.push(pos[comptime!(self.start_axis + i)] % tile_size);
        }

        let post_start = comptime!(self.start_axis + n);
        let physical_post_start = comptime!(self.start_axis + 2 * n);
        #[unroll]
        for i in 0..comptime!(physical_rank - physical_post_start) {
            physical.push(pos[comptime!(post_start + i)]);
        }

        physical
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        #[comptime]
        let n = self.tiles.len();
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut semantic = CoordsDyn::new();

        #[unroll]
        for i in 0..self.start_axis {
            semantic.push(self.physical_shape[i]);
        }

        #[unroll]
        for i in 0..n {
            let grid = self.physical_shape[comptime!(self.start_axis + i)];
            let tile = self.physical_shape[comptime!(self.start_axis + n + i)];
            semantic.push(grid * tile);
        }

        let post_start = comptime!(self.start_axis + 2 * n);
        #[unroll]
        for i in 0..comptime!(physical_rank - post_start) {
            semantic.push(self.physical_shape[comptime!(post_start + i)]);
        }

        semantic
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut is_valid = true;
        #[unroll]
        for i in 0..bounds.len() {
            is_valid = is_valid && pos[i] < bounds[i];
        }
        is_valid
    }
}

/// Maps multi-dimensional physical coordinates to a linear buffer offset using
/// per-axis strides. The physical layout's rank can be any value.
#[derive(CubeType, Clone)]
pub struct StridedNdLayout {
    shape: CoordsDyn,
    strides: CoordsDyn,
}

#[cube]
impl StridedNdLayout {
    pub fn new(shape: CoordsDyn, strides: CoordsDyn) -> StridedNdLayout {
        StridedNdLayout { shape, strides }
    }
}

#[cube]
impl Layout for StridedNdLayout {
    type Coordinates = CoordsDyn;

    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut offset = 0u32;
        #[unroll]
        for i in 0..self.strides.len() {
            offset += pos[i] * self.strides[i];
        }
        offset as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos.clone()), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut is_valid = true;
        #[unroll]
        for i in 0..self.shape.len() {
            is_valid = is_valid && pos[i] < self.shape[i];
        }
        is_valid
    }
}
