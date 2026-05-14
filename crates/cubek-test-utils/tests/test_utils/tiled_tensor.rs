use cubecl::{
    self,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
    zspace::{Shape, SmallVec, Strides},
};
use cubecl::{TestRuntime, prelude::*};
use cubecl::{
    std::tensor::{AsView, AsViewExpand, AsViewMut, AsViewMutExpand},
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx};

#[test]
fn read_rowmajor_tensor_as_tiled_layout_v2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let matrix_len = 4;
    let shape = shape![matrix_len, matrix_len];

    let input_handle = TestInput::builder(client.clone(), shape.clone())
        .stride(StrideSpec::RowMajor)
        .arange()
        .generate();

    let dtype = f32::as_type_native_unchecked().storage_type();
    let output_handle = TestInput::builder(client.clone(), shape.clone())
        .stride(StrideSpec::RowMajor)
        .zeros()
        .generate_without_host_data();

    let metadata = input_handle.metadata.to_tiled(0, &[2, 2]);
    let blueprint = MetadataBlueprint::new(
        metadata.shape.clone(),
        metadata.strides.clone(),
        metadata.tiler.map(|t| TilerBlueprint {
            start_axis: t.start_axis,
            tile_size: t.tile_size,
        }),
    );

    let cube_count = CubeCount::new_single();
    let cube_dim = CubeDim::new_single();
    let vector_size = 1;

    launch_read_tensor_as_tiled::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        blueprint,
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

    let (_, expected_values) = TestInput::builder(client, shape)
        .custom(expected_values)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected_values, 1e-6)
        .as_test_outcome()
        .enforce()
}

#[test]
fn read_rowmajor_tensor_as_tiled() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let matrix_len = 4;
    let shape = shape![matrix_len, matrix_len];
    let input_handle = TestInput::builder(client.clone(), shape.clone())
        .stride(StrideSpec::RowMajor)
        .arange()
        .generate();

    let dtype = f32::as_type_native_unchecked().storage_type();
    let output_handle = TestInput::builder(client.clone(), shape.clone())
        .stride(StrideSpec::RowMajor)
        .zeros()
        .generate_without_host_data();

    let cube_count = CubeCount::new_single();
    let cube_dim = CubeDim::new_single();
    let vector_size = 1;
    launch_read_rowmajor_tensor_as_tiled::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        dtype,
        vector_size,
        matrix_len,
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

    #[rustfmt::skip]
    let expected_values = [
        0.000,  1.000,  4.000,  5.000,
        2.000,  3.000,  6.000,  7.000,
        8.000,  9.000, 12.000, 13.000,
        10.00, 11.000, 14.000, 15.000,
    ].to_vec();
    let (_, expected_values) = TestInput::builder(client, shape)
        .custom(expected_values)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected_values, 1e-6)
        .as_test_outcome()
        .enforce()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TilerBlueprint {
    pub start_axis: u8,
    pub tile_size: SmallVec<[u16; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetadataBlueprint {
    shape: Shape,
    strides: Strides,
    tiler: Option<TilerBlueprint>,
}

impl MetadataBlueprint {
    pub fn new(shape: Shape, strides: Strides, tiler: Option<TilerBlueprint>) -> Self {
        MetadataBlueprint {
            shape,
            strides,
            tiler,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct TiledLayout {
    width: usize,
    height: usize,
    tile_w: usize,
    tile_h: usize,
}

#[cube]
impl TiledLayout {
    pub fn new(width: usize, height: usize, tile_w: usize, tile_h: usize) -> TiledLayout {
        TiledLayout {
            width,
            height,
            tile_w,
            tile_h,
        }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = (usize, usize);

    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let row = pos.0;
        let col = pos.1;

        let tile_row = row / self.tile_h;
        let tile_col = col / self.tile_w;

        let local_row = row % self.tile_h;
        let local_col = col % self.tile_w;

        let tiles_per_row = self.width / self.tile_w;

        let tile_index = tile_row * tiles_per_row + tile_col;
        let tile_area = self.tile_h * self.tile_w;

        let block_offset = tile_index * tile_area;
        let local_offset = local_row * self.tile_w + local_col;

        block_offset + local_offset
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
fn launch_read_rowmajor_tensor_as_tiled<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: &mut Tensor<Vector<N, S>>,
    #[define(N)] _dtype: StorageType,
    #[define(S)] vector_size: usize,
    #[comptime] matrix_len: usize,
) {
    let input_view = input.view(TiledLayout::new(
        matrix_len,
        matrix_len,
        matrix_len / 2,
        matrix_len / 2,
    ));
    let output_view = output.view_mut(RowMajorLayout::new(matrix_len, matrix_len, vector_size));

    for i in 0..matrix_len {
        for j in 0..matrix_len {
            let value = input_view.read((i, j));
            output_view.write((i, j), value);
        }
    }
}

#[cube(launch)]
fn launch_read_tensor_as_tiled<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: &mut Tensor<Vector<N, S>>,
    #[comptime] blueprint: MetadataBlueprint,
    #[comptime] matrix_len: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] vector_size: usize,
) {
    let tiler = blueprint.tiler.clone().unwrap();

    let mut shape = Sequence::new();
    #[unroll]
    for i in 0..blueprint.shape.rank() {
        shape.push(comptime!(blueprint.shape[i]));
    }

    let mut strides = Sequence::new();

    #[unroll]
    for i in 0..blueprint.strides.rank() {
        strides.push(comptime!(blueprint.strides[i]));
    }

    let mut tiles = Sequence::new();
    #[unroll]
    for i in 0..tiler.tile_size.len() {
        tiles.push(comptime!(tiler.tile_size[i]) as usize);
    }

    let tiled_layout = TiledLayoutV2::new(shape, strides, tiler.start_axis as usize, tiles);

    let row_major = RowMajorLayout::new(4, 4, vector_size);

    let input_view = input.view(tiled_layout);
    let output_view = output.view_mut(row_major);

    #[unroll]
    for i in 0..matrix_len {
        #[unroll]
        for j in 0..matrix_len {
            let mut coords = Sequence::<usize>::new();
            coords.push(i);
            coords.push(j);
            let value = input_view.read(coords);
            output_view.write((i.runtime(), j.runtime()), value);
        }
    }
}

#[derive(CubeType, Clone)]
pub struct TiledLayoutV2 {
    shape: Sequence<usize>,
    strides: Sequence<usize>,
    #[cube(comptime)]
    start_axis: usize,
    tiles: Sequence<usize>,
}

#[cube]
impl TiledLayoutV2 {
    pub fn new(
        shape: Sequence<usize>,
        strides: Sequence<usize>,
        #[comptime] start_axis: usize,
        tiles: Sequence<usize>,
    ) -> TiledLayoutV2 {
        TiledLayoutV2 {
            shape,
            strides,
            start_axis,
            tiles,
        }
    }
}

#[cube]
impl Layout for TiledLayoutV2 {
    type Coordinates = Sequence<usize>;

    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut offset = 0;
        #[comptime]
        let s = self.start_axis;
        #[comptime]
        let n = self.tiles.len();
        let logical_rank = pos.len();

        #[unroll]
        for i in 0..s {
            offset += pos[i] * self.strides[i];
        }

        #[unroll]
        for i in 0..n {
            let logical_idx = comptime!(self.start_axis + i);
            let tile_size = self.tiles[i];

            let grid_coord = pos[logical_idx] / tile_size;
            let local_coord = pos[logical_idx] % tile_size;

            offset += grid_coord * self.strides[logical_idx];
            offset += local_coord * self.strides[comptime!(logical_idx + n)];
        }
        let start = comptime!(self.start_axis + n);
        #[unroll]
        for i in start..logical_rank {
            offset += pos[i] * self.strides[comptime!(i + n)];
        }

        offset
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let is_valid = pos[0] < self.shape[0] && pos[1] < self.shape[1];
        (self.to_source_pos(pos), is_valid)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }
}
