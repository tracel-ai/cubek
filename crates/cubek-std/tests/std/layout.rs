use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand,
        layout::{
            CoordsDyn,
            semantic_tensor::{SemanticTensorView, semantic_tensor_view},
            tiled_tensor::{TiledTensorView, tiled_tensor_view},
        },
    },
    zspace::shape,
};
use cubek_std::layout::RowMajorLayout;
use cubek_test_utils::{
    HostData, HostDataType, LayoutSpec, StridedLayout, TestInput, assert_equals_approx,
};

/// Kernel writes via semantic (M, N) coords; output buffer is tiled — the
/// layout transparently routes each write to its tile-permuted offset.
#[test]
fn write_via_semantic_view_to_tiled_output() {
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

    // `semantic_tensor_view` auto-derives the semantic-coordinate view from
    // the tensor's metadata — the kernel never sees the tile grid.
    launch_write_via_semantic_view::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        semantic_tensor_view(output_handle.clone().binding()),
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

/// Semantic-view kernel against a plain (no-Tiler) output. The layout
/// collapses to a strided view.
#[test]
fn write_via_semantic_view_to_non_tiled_output() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let matrix_len = 4;
    let shape = shape![matrix_len, matrix_len];

    let input_handle = TestInput::builder(client.clone(), shape.clone())
        .arange()
        .generate();

    let dtype = f32::as_type_native_unchecked().storage_type();

    // No tiled layout — just plain row-major. No Tiler on the metadata.
    let output_handle = TestInput::builder(client.clone(), shape.clone())
        .zeros()
        .generate_without_host_data();

    let cube_count = CubeCount::new_single();
    let cube_dim = CubeDim::new_single();
    let vector_size = 1;

    launch_write_via_semantic_view::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        semantic_tensor_view(output_handle.clone().binding()),
        matrix_len,
        dtype,
        vector_size,
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

    // With no tiling, output should match input arange order exactly.
    let (_, expected_values) = TestInput::builder(client, shape)
        .arange()
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected_values, 1e-6)
        .as_test_outcome()
        .enforce()
}

/// Kernel addresses the tensor in tile-grained coords (grid outer, tile
/// inner).
#[test]
fn write_via_tiled_view_to_tiled_output() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let matrix_len = 4;
    let tile_size = 2;
    let grid_len = matrix_len / tile_size;
    let shape = shape![matrix_len, matrix_len];

    let input_handle = TestInput::builder(client.clone(), shape.clone())
        .arange()
        .generate();

    let dtype = f32::as_type_native_unchecked().storage_type();

    let tiled_layout = LayoutSpec::tiled(StridedLayout::RowMajor, 0, vec![2u16, 2]);
    let output_handle = TestInput::builder(client.clone(), shape.clone())
        .layout(tiled_layout.clone())
        .zeros()
        .generate_without_host_data();

    let cube_count = CubeCount::new_single();
    let cube_dim = CubeDim::new_single();
    let vector_size = 1;

    // `tiled_tensor_view` exposes the rank-expanded physical coords — the
    // kernel walks the tile grid explicitly (loader-style).
    launch_write_via_tiled_view::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        input_handle.binding().into_tensor_arg(),
        tiled_tensor_view(output_handle.clone().binding()),
        matrix_len,
        tile_size,
        grid_len,
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
        .layout(tiled_layout)
        .custom(expected_values)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected_values, 1e-6)
        .as_test_outcome()
        .enforce()
}

#[cube(launch)]
fn launch_write_via_semantic_view<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: SemanticTensorView<Vector<N, S>, ReadWrite>,
    #[comptime] matrix_len: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] vector_size: usize,
) {
    let row_major = RowMajorLayout::new(matrix_len, matrix_len, vector_size);
    let input_view = input.view(row_major);

    #[unroll]
    for i in 0..matrix_len {
        #[unroll]
        for j in 0..matrix_len {
            let mut coords = CoordsDyn::new();
            coords.push(i as u32);
            coords.push(j as u32);
            let value = input_view.read((i.runtime(), j.runtime()));
            output.write(coords, value);
        }
    }
}

#[cube(launch)]
fn launch_write_via_tiled_view<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: TiledTensorView<Vector<N, S>, ReadWrite>,
    #[comptime] matrix_len: usize,
    #[comptime] tile_size: usize,
    #[comptime] grid_len: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] vector_size: usize,
) {
    let row_major = RowMajorLayout::new(matrix_len, matrix_len, vector_size);
    let input_view = input.view(row_major);

    // Walk the tile grid explicitly — this is the loader pattern.
    #[unroll]
    for grid_m in 0..grid_len {
        #[unroll]
        for grid_n in 0..grid_len {
            #[unroll]
            for tile_m in 0..tile_size {
                #[unroll]
                for tile_n in 0..tile_size {
                    let sem_i = grid_m * tile_size + tile_m;
                    let sem_j = grid_n * tile_size + tile_n;
                    let value = input_view.read((sem_i.runtime(), sem_j.runtime()));

                    let mut coords = CoordsDyn::new();
                    coords.push(grid_m as u32);
                    coords.push(grid_n as u32);
                    coords.push(tile_m as u32);
                    coords.push(tile_n as u32);
                    output.write(coords, value);
                }
            }
        }
    }
}
