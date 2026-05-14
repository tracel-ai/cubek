use cubecl::{
    self,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
    zspace::metadata::Metadata,
};
use cubecl::{TestRuntime, prelude::*};
use cubecl::{
    std::tensor::{AsView, AsViewExpand, AsViewMut, AsViewMutExpand},
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx};

#[test]
fn read_rowmajor_tensor_as_tiled_layout() {
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

    let cube_count = CubeCount::new_single();
    let cube_dim = CubeDim::new_single();
    let vector_size = 1;

    launch_read_tensor_as_tiled::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        metadata,
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

    let mut shape = Sequence::new();
    #[unroll]
    for i in 0..metadata.shape.rank() {
        shape.push(comptime!(metadata.shape[i]));
    }

    let mut strides = Sequence::new();

    #[unroll]
    for i in 0..metadata.strides.rank() {
        strides.push(comptime!(metadata.strides[i]));
    }

    let mut tiles = Sequence::new();
    #[unroll]
    for i in 0..tiler.tile_size.len() {
        tiles.push(comptime!(tiler.tile_size[i]) as usize);
    }

    let tiled_layout = TiledLayout::new(shape, strides, tiler.start_axis as usize, tiles);

    let row_major = RowMajorLayout::new(matrix_len, matrix_len, vector_size);

    let input_view = input.view(row_major);
    let output_view = output.view_mut(tiled_layout);

    #[unroll]
    for i in 0..matrix_len {
        #[unroll]
        for j in 0..matrix_len {
            let mut coords = Sequence::<usize>::new();
            coords.push(i);
            coords.push(j);
            let value = input_view.read((i.runtime(), j.runtime()));
            output_view.write(coords, value);
        }
    }
}

#[derive(CubeType, Clone)]
pub struct TiledLayout {
    shape: Sequence<usize>,
    strides: Sequence<usize>,
    #[cube(comptime)]
    start_axis: usize,
    tiles: Sequence<usize>,
}

#[cube]
impl TiledLayout {
    pub fn new(
        shape: Sequence<usize>,
        strides: Sequence<usize>,
        #[comptime] start_axis: usize,
        tiles: Sequence<usize>,
    ) -> TiledLayout {
        TiledLayout {
            shape,
            strides,
            start_axis,
            tiles,
        }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = Sequence<usize>;

    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut offset = 0;
        #[comptime]
        let n = self.tiles.len();
        let rank = pos.len();

        #[unroll]
        for i in 0..self.start_axis {
            offset += pos[i] * self.strides[i];
        }

        #[unroll]
        for i in 0..n {
            let physical_idx = comptime!(self.start_axis + i);
            let tile_size = self.tiles[i];

            let grid_coord = pos[physical_idx] / tile_size;
            let local_coord = pos[physical_idx] % tile_size;

            offset += grid_coord * self.strides[physical_idx];
            offset += local_coord * self.strides[comptime!(physical_idx + n)];
        }

        let start = comptime!(self.start_axis + n);
        #[unroll]
        for i in start..rank {
            offset += pos[i] * self.strides[comptime!(i + n)];
        }

        offset
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
            is_valid = is_valid && pos[i] >= self.shape[i];
        }
        is_valid
    }
}
