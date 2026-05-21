use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand,
        layout::{CoordsDyn, chain::Chain},
    },
    zspace::{metadata::Metadata, shape},
};
use cubek_std::layout::{DynamicRankStridedLayout, RowMajorLayout, TiledLayout};
use cubek_test_utils::{
    HostData, HostDataType, LayoutSpec, StridedLayout, TestInput, assert_equals_approx,
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
    let tiled_layout = LayoutSpec::tiled(StridedLayout::RowMajor, 0, vec![2u16, 2]);
    let output_handle = TestInput::builder(client.clone(), shape.clone())
        .layout(tiled_layout.clone())
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
        .layout(tiled_layout)
        .custom(expected_values)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected_values, 1e-6)
        .as_test_outcome()
        .enforce()
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
    let physical_to_buffer = DynamicRankStridedLayout::new(physical_shape, physical_strides);
    let tiled_layout = Chain::<DynamicRankStridedLayout, TiledLayout>::new(
        physical_to_buffer,
        semantic_to_physical,
    );

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
