use cubecl::{Runtime, TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TileInput, assert_equals_approx,
};
use cubek_tile::{
    Axis, ByAxis, ComputeScope, Coverage, CubeAxis, Distribution, Partitioner, Space, Spread,
    TileArgLaunch,
};

const SEED: u64 = 0x1;

#[test]
fn dequantize_tiled_native_per_tensor_matches_reference() {
    // `&[]` -> one block covering the whole tensor.
    dequantize_tiled_native(&[8, 8], 4, &[]);
}

/// Launch the tiled dequantize and assert it matches the CPU reference.
fn dequantize_tiled_native(tensor: &[usize], tile_edge: usize, block: &[usize]) {
    let client = TestRuntime::client(&Default::default());
    let rank = tensor.len();
    let per_tensor = block.is_empty();
    let block_dims = block_dims(tensor, block);
    let grid: Vec<usize> = tensor
        .iter()
        .zip(&block_dims)
        .map(|(&t, &b)| t / b)
        .collect();
    let scale_shape = if per_tensor {
        vec![1usize]
    } else {
        grid.clone()
    };

    let tile_edges = vec![tile_edge; rank];
    let values = TileInput::builder(&client, spatial_space(tensor, &tile_edges))
        .untiled()
        .arange();
    let scales = TileInput::builder(&client, spatial_space(&scale_shape, &scale_shape))
        .untiled()
        .uniform(SEED, 0.25, 2.0);
    let output = TileInput::builder(&client, spatial_space(tensor, &tile_edges))
        .untiled()
        .zeros();

    let values_space = values.space();
    let cube_count = values_space.partitioner().cube_count(&values_space);
    let cube_dim = values_space.partitioner().cube_dim(&client, &values_space);

    let dtype = f32::as_type_native_unchecked().storage_type();
    cubek_quant::dequantize_tiled::dequantize::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        TileArgLaunch::new(values.tensor_arg(1), values.space(), values.storage()),
        TileArgLaunch::new(scales.tensor_arg(1), scales.space(), scales.storage()),
        TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
        dtype,
        dtype,
        dtype,
        1usize,
        1usize,
        1usize,
    );

    check_output(
        &client,
        scales.handle(),
        output.handle(),
        tensor,
        &block_dims,
        per_tensor,
    );
}

/// Check that the tiled dequantize output matches the CPU reference `values[i] * scales[cell_of(i)]`.
fn check_output(
    client: &ComputeClient<TestRuntime>,
    scales_h: TensorHandle<TestRuntime>,
    output_h: TensorHandle<TestRuntime>,
    tensor: &[usize],
    block_dims: &[usize],
    per_tensor: bool,
) {
    let scales = HostData::from_tensor_handle(client, scales_h, HostDataType::F32);
    let got = HostData::from_tensor_handle(client, output_h, HostDataType::F32);

    let shape = Shape::from(tensor.to_vec());
    let expected = HostData {
        data: HostDataVec::F32(
            got.iter_indices()
                .enumerate()
                .map(|(i, coord)| {
                    let scale_coord = if per_tensor {
                        vec![0usize]
                    } else {
                        coord.iter().zip(block_dims).map(|(&c, &b)| c / b).collect()
                    };
                    i as f32 * scales.get_f32(&scale_coord)
                })
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };
    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}

/// Per-axis block shape: `&[]` -> the whole tensor (per-tensor); otherwise the block is
/// trailing-aligned and leading axes get `1`.
fn block_dims(tensor: &[usize], block: &[usize]) -> Vec<usize> {
    if block.is_empty() {
        return tensor.to_vec();
    }
    assert!(
        block.len() <= tensor.len(),
        "block rank exceeds tensor rank"
    );
    let mut dims = vec![1usize; tensor.len()];
    dims[tensor.len() - block.len()..].copy_from_slice(block);
    dims
}

/// A `Space` with one cube per tile: each axis maps to a cube dimension (X/Y/Z) and
/// carries exactly one tile. Rank must be ≤ 3.
fn spatial_space(extents: &[usize], tile_edges: &[usize]) -> Space {
    const CUBE_AXES: [CubeAxis; 3] = [CubeAxis::X, CubeAxis::Y, CubeAxis::Z];
    let rank = extents.len();
    assert!(rank <= CUBE_AXES.len(), "spatial tiling supports rank ≤ 3");
    let axes: Vec<Axis> = (0..rank as u8).map(Axis).collect();
    let edges: Vec<(Axis, usize)> = axes
        .iter()
        .copied()
        .zip(tile_edges.iter().copied())
        .collect();
    let dists: Vec<(Axis, Distribution)> = axes
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            (
                a,
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CUBE_AXES[i]),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                },
            )
        })
        .collect();
    let entries: Vec<(Axis, usize)> = axes.iter().copied().zip(extents.iter().copied()).collect();
    Space::new(&entries)
        .with_partitioner(Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists)).direct())
}
