//! Correctness harness for the tiled dequantization kernel
//! (`Tile::dequantize2`, launched via `cubek_quant::dequantize_tiled::dequantize`).
//!
//! Native (unpacked) storage: quantized values stand in as plain `f32`, so the tests
//! exercise the tile plumbing (walk → per-tile access → block→scale lookup) rather than
//! integer unpacking. [`check_output`] runs the kernel for a given tensor shape, tile
//! edge and block shape, and diffs against a CPU reference; per-tensor / 1-D / 2-D / 3-D
//! blocks are just different arguments.

use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::Shape,
};
use cubek_tile::{Axis, ByAxis, Distribution, Partitioner, Space, Storage, TileArgLaunch};

const SEED: u64 = 0xC0FF_EE00;

#[test]
fn dequantize_tiled_native_per_tensor_matches_reference() {
    // `&[]` -> one block covering the whole tensor.
    check_output(&[8, 8], 4, &[]);
}

#[test]
fn dequantize_tiled_native_per_block_1d_matches_reference() {
    // `&[4]` -> a 1-D block over the last axis ([1,4] in 2-D terms).
    check_output(&[8, 8], 4, &[4]);
}

#[test]
fn dequantize_tiled_native_per_block_2d_matches_reference() {
    // `&[4,4]` -> a 2-D block.
    check_output(&[8, 8], 4, &[4, 4]);
}

#[test]
fn dequantize_tiled_native_per_block_3d_matches_reference() {
    // `&[2,2,2]` -> a 3-D block over a rank-3 tensor, tiled 2×2×2.
    check_output(&[4, 4, 4], 2, &[2, 2, 2]);
}

/// Launch the tiled dequantize and assert it matches the CPU reference
/// `values[i] * scales[cell_of(i)]`, for any rank.
///
/// - `tensor`: the logical shape (any rank).
/// - `tile_edge`: every axis is tiled into squares of this edge (must divide each dim).
/// - `block`: per-axis block shape, trailing-aligned like `BlockSize`. `&[]` is
///   per-tensor (one block); a shorter slice left-pads with `1` (each leading axis is its
///   own block). The scale grid is one scale per block: `tensor[a] / block[a]` per axis.
fn check_output(tensor: &[usize], tile_edge: usize, block: &[usize]) {
    let client = TestRuntime::client(&Default::default());
    let rank = tensor.len();
    let n_elems: usize = tensor.iter().product();

    let block_dims = block_dims(tensor, block);
    let grid: Vec<usize> = tensor
        .iter()
        .zip(&block_dims)
        .map(|(&t, &b)| t / b)
        .collect();
    let n_scales: usize = grid.iter().product();

    // Values: arange. Scales: distinct per block, so the mapping is actually tested.
    let values: Vec<f32> = (0..n_elems).map(|i| i as f32).collect();
    let scales = random_scales(SEED, n_scales, 0.25, 2.0);

    let ax = axes(rank);
    let tile_edges = vec![tile_edge; rank];

    let values_h = tensor_nd(&client, &values, tensor);
    let scales_h = tensor_nd(&client, &scales, &grid);
    let output_h = TensorHandle::<TestRuntime>::zeros(
        &client,
        Shape::from(tensor.to_vec()),
        f32::as_type_native_unchecked(),
    );

    // Values/output tile into `tile_edge` squares; scales are the block grid itself
    // (its partitioner is unused — dequantize2 reads scales by absolute cell coord).
    let values_space = space_nd(&ax, tensor, &tile_edges);
    let output_space = space_nd(&ax, tensor, &tile_edges);
    let scales_space = space_nd(&ax, &grid, &grid);
    // Plain row-major buffers: physical rank == logical rank, so 0 tile levels.
    let storage = Storage::of(rank, rank);

    let dtype = f32::as_type_native_unchecked().storage_type();

    cubek_quant::dequantize_tiled::dequantize::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(values_h.binding().into_tensor_arg(), values_space, storage),
        TileArgLaunch::new(scales_h.binding().into_tensor_arg(), scales_space, storage),
        TileArgLaunch::new(
            output_h.clone().binding().into_tensor_arg(),
            output_space,
            storage,
        ),
        dtype,
        dtype,
        dtype,
        1usize,
        1usize,
        1usize,
    );

    let bytes = client.read_one(output_h.handle).unwrap();
    let got = f32::from_bytes(&bytes);

    // Reference: element i's grid cell is coord[a] / block[a] per axis, raveled row-major.
    let expected: Vec<f32> = (0..n_elems)
        .map(|i| {
            let coord = unravel(i, tensor);
            let cell: Vec<usize> = coord
                .iter()
                .zip(&block_dims)
                .map(|(&c, &b)| c / b)
                .collect();
            values[i] * scales[ravel(&cell, &grid)]
        })
        .collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "mismatch at {i}: got {g}, expected {e}"
        );
    }
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

fn tensor_nd(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    data: &[f32],
    shape: &[usize],
) -> TensorHandle<TestRuntime> {
    let shape = Shape::from(shape.to_vec());
    let alloc =
        client.create_tensor_from_slice(f32::as_bytes(data), shape.clone(), f32::type_size());
    TensorHandle::new(
        alloc.memory,
        shape,
        alloc.strides,
        f32::as_type_native_unchecked(),
    )
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

/// Deterministic pseudo-random `f32`s in `[lo, hi)` via SplitMix64 — no `rand` dep, and
/// reproducible so a failing case always reproduces.
fn random_scales(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Top 24 bits -> a uniform float in [0, 1).
            let unit = (z >> 40) as f32 / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}
