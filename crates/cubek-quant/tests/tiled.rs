//! Correctness harness for the tiled dequantization kernel
//! (`Tile::dequantize3`, launched via `cubek_quant::dequantize_tiled::dequantize`).
//!
//! Native (unpacked) storage: quantized values stand in as plain `f32`, so the tests
//! exercise the tile plumbing (walk → per-tile access → block→scale lookup) rather than
//! integer unpacking. [`check_output`] runs the kernel for a given tensor shape, tile
//! edge and block shape, and diffs against a CPU reference; per-tensor / 1-D / 2-D / 3-D
//! blocks are just different arguments. Operand data and the comparison come from
//! `cubek-test-utils`; only the cubek-tile `Space`/`Partitioner` wiring is built here.

use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::Shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubek_tile::{Axis, ByAxis, Distribution, Partitioner, Space, TileArgLaunch};

const SEED: u64 = 0xC0FF_EE00;

#[test]
fn dequantize_tiled_native_per_tensor_matches_reference() {
    // `&[]` -> one block covering the whole tensor.
    dequantize_tiled_native(&[8, 8], 4, &[]);
}

#[test]
fn dequantize_tiled_native_per_block_1d_matches_reference() {
    // `&[4]` -> a 1-D block over the last axis ([1,4] in 2-D terms).
    dequantize_tiled_native(&[8, 8], 4, &[4]);
}

#[test]
fn dequantize_tiled_native_per_block_2d_matches_reference() {
    // `&[4,4]` -> a 2-D block.
    dequantize_tiled_native(&[8, 8], 4, &[4, 4]);
}

#[test]
fn dequantize_tiled_native_per_block_3d_matches_reference() {
    // `&[2,2,2]` -> a 3-D block over a rank-3 tensor, tiled 2×2×2.
    dequantize_tiled_native(&[4, 4, 4], 2, &[2, 2, 2]);
}

/// Launch the tiled dequantize and assert it matches the CPU reference
/// `values[i] * scales[cell_of(i)]`, for any rank.
///
/// - `tensor`: the logical shape (any rank).
/// - `tile_edge`: every axis is tiled into squares of this edge (must divide each dim).
/// - `block`: per-axis block shape, trailing-aligned like `BlockSize`. `&[]` is
///   per-tensor (one block); a shorter slice left-pads with `1` (each leading axis is its
///   own block). The scale grid is one scale per block: `tensor[a] / block[a]` per axis.
fn dequantize_tiled_native(tensor: &[usize], tile_edge: usize, block: &[usize]) {
    let client = TestRuntime::client(&Default::default());
    let rank = tensor.len();

    let block_dims = block_dims(tensor, block);
    let grid: Vec<usize> = tensor
        .iter()
        .zip(&block_dims)
        .map(|(&t, &b)| t / b)
        .collect();

    // cubek-tile wiring: tile values/output into `tile_edge` squares; scales are the
    // block grid itself (its partitioner is unused — dequantize3 reads scales by
    // absolute cell coord). Plain row-major buffers, so 0 tile levels.
    let ax = axes(rank);
    let tile_edges = vec![tile_edge; rank];
    let values = TileInput::builder(&client, space_nd(&ax, tensor, &tile_edges))
        .untiled()
        .arange();
    let scales = TileInput::builder(&client, space_nd(&ax, &grid, &grid))
        .untiled()
        .uniform(SEED, 0.25, 2.0);
    let output = TileInput::builder(&client, space_nd(&ax, tensor, &tile_edges))
        .untiled()
        .zeros();

    let dtype = f32::as_type_native_unchecked().storage_type();
    cubek_quant::dequantize_tiled::dequantize::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
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

    // Reference: values are arange, so element i's value is `i`; its block cell is
    // coord[a] / block[a] per axis, raveled row-major into the scale grid.
    check_output(
        &client,
        scales.handle(),
        output.handle(),
        tensor,
        &grid,
        &block_dims,
    );
}

fn check_output(
    client: &ComputeClient<TestRuntime>,
    scales_h: TensorHandle<TestRuntime>,
    output_h: TensorHandle<TestRuntime>,
    tensor: &[usize],
    grid: &[usize],
    block_dims: &[usize],
) {
    let scales_host = HostData::from_tensor_handle(&client, scales_h.clone(), HostDataType::F32);
    let n_elems: usize = tensor.iter().product();

    let expected: Vec<f32> = (0..n_elems)
        .map(|i| {
            let coord = unravel(i, tensor);
            let cell: Vec<usize> = coord.iter().zip(block_dims).map(|(&c, &b)| c / b).collect();
            i as f32 * scales_host.data.get_f32(ravel(&cell, grid))
        })
        .collect();

    let got = HostData::from_tensor_handle(&client, output_h, HostDataType::F32);
    let (_, expected) = TestInput::builder(client.clone(), Shape::from(tensor.to_vec()))
        .custom(expected)
        .generate_with_f32_host_data();
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

/// The first `rank` canonical axes.
fn axes(rank: usize) -> Vec<Axis> {
    (0..rank as u8).map(Axis).collect()
}

/// A row-major, single-instance partitioner tiling `axes` by the given per-axis edges.
fn row_major_seq_nd(axes: &[Axis], edges: &[usize]) -> Partitioner {
    let edges: Vec<(Axis, usize)> = axes.iter().copied().zip(edges.iter().copied()).collect();
    let dists: Vec<(Axis, Distribution)> = axes
        .iter()
        .map(|&a| (a, Distribution::Sequential))
        .collect();
    Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists)).direct()
}

/// A `Space` over `axes` with the given extents, tiled by `edges`.
fn space_nd(axes: &[Axis], extents: &[usize], edges: &[usize]) -> Space {
    let entries: Vec<(Axis, usize)> = axes.iter().copied().zip(extents.iter().copied()).collect();
    Space::new(&entries).with_partitioner(row_major_seq_nd(axes, edges))
}

/// Row-major unravel of a flat index into per-axis coordinates.
fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut coord = vec![0usize; shape.len()];
    for a in (0..shape.len()).rev() {
        coord[a] = i % shape[a];
        i /= shape[a];
    }
    coord
}

/// Row-major ravel of per-axis coordinates back into a flat index.
fn ravel(coord: &[usize], shape: &[usize]) -> usize {
    coord.iter().zip(shape).fold(0, |idx, (&c, &s)| idx * s + c)
}
