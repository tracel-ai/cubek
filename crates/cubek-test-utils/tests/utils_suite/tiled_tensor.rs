use cubecl::{std::tensor::{AsView, AsViewExpand, AsViewMut, AsViewMutExpand}, zspace::shape};
use cubecl::{
    self,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
};
use cubecl::{TestRuntime, prelude::*};
use cubek_test_utils::{
    HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx,
};


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
