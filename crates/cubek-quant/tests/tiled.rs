//! Correctness harness for the tiled dequantization kernel
//! (`Tile::dequantize2`, launched via `cubek_quant::dequantize_tiled::dequantize`).
//!
//! Native (unpacked) storage: quantized values stand in as plain `f32`, so the tests
//! exercise the tile plumbing (walk → per-tile access → block→scale lookup) rather than
//! integer unpacking. [`check_output`] runs the kernel for a given block shape and diffs
//! against a CPU reference; the per-tensor / 1-D / 2-D cases are just different block
//! shapes.

use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::shape,
};
use cubek_tile::{Axis, ByAxis, Distribution, Partitioner, Space, Storage, TileArgLaunch};

const M: Axis = Axis(0);
const N: Axis = Axis(1);

const SEED: u64 = 0xC0FF_EE00;
const M_SIZE: usize = 8;
const N_SIZE: usize = 8;
const TILE_EDGE: usize = 4;

#[test]
fn dequantize_tiled_native_per_tensor_matches_reference() {
    // `&[]` -> one block covering the whole tensor.
    check_output(&[]);
}

#[test]
fn dequantize_tiled_native_per_block_1d_matches_reference() {
    // `&[4]` -> a 1-D block over the last axis ([1,4] in 2-D terms).
    check_output(&[4]);
}

#[test]
fn dequantize_tiled_native_per_block_2d_matches_reference() {
    // `&[4,4]` -> a 2-D block.
    check_output(&[4, 4]);
}

/// Launch the tiled dequantize for a given block shape and assert it matches the CPU
/// reference `values[i] * scales[block_of(i)]`.
///
/// `block` is the per-axis block shape, trailing-aligned like `BlockSize`:
/// - `&[]` — per-tensor (a single block),
/// - `&[bn]` — 1-D block over the last axis,
/// - `&[bm, bn]` — 2-D block.
fn check_output(block: &[usize]) {
    let client = TestRuntime::client(&Default::default());
    let (m, n) = (M_SIZE, N_SIZE);

    // Block shape in 2-D terms; the scale grid is one scale per block.
    let (block_m, block_n) = match block {
        [] => (m, n),
        [bn] => (1, *bn),
        [bm, bn] => (*bm, *bn),
        _ => panic!("block must have rank 0, 1, or 2, got {block:?}"),
    };
    let (grid_m, grid_n) = (m / block_m, n / block_n);

    // Values: arange. Scales: distinct per block (so the mapping is actually tested).
    let values: Vec<f32> = (0..(m * n)).map(|i| i as f32).collect();
    let scales = random_scales(SEED, grid_m * grid_n, 0.25, 2.0);

    let values_h = tensor_from(&client, &values, m, n);
    let scales_h = tensor_from(&client, &scales, grid_m, grid_n);
    let output_h =
        TensorHandle::<TestRuntime>::zeros(&client, shape![m, n], f32::as_type_native_unchecked());

    // Values/output tile into `TILE_EDGE` squares; scales are the block grid itself.
    let values_space =
        Space::new(&[(M, m), (N, n)]).with_partitioner(row_major_seq(TILE_EDGE, TILE_EDGE));
    let output_space =
        Space::new(&[(M, m), (N, n)]).with_partitioner(row_major_seq(TILE_EDGE, TILE_EDGE));
    let scales_space =
        Space::new(&[(M, grid_m), (N, grid_n)]).with_partitioner(row_major_seq(grid_m, grid_n));
    // Plain row-major buffers: physical rank == logical rank, so 0 tile levels.
    let storage = Storage::of(2, 2);

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

    // Block grid, row-major: element (r, c) belongs to grid cell (r/block_m, c/block_n).
    let expected: Vec<f32> = (0..(m * n))
        .map(|i| {
            let (r, c) = (i / n, i % n);
            let cell = (r / block_m) * grid_n + (c / block_n);
            values[i] * scales[cell]
        })
        .collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "mismatch at {i}: got {g}, expected {e}"
        );
    }
}

/// A row-major, single-instance partitioner tiling the `(M, N)` axes by the given edges.
fn row_major_seq(edge_m: usize, edge_n: usize) -> Partitioner {
    Partitioner::row_major(
        ByAxis::new(&[(M, edge_m), (N, edge_n)]),
        ByAxis::new(&[(M, Distribution::Sequential), (N, Distribution::Sequential)]),
    )
    .direct()
}

fn tensor_from(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> TensorHandle<TestRuntime> {
    let alloc =
        client.create_tensor_from_slice(f32::as_bytes(data), shape![rows, cols], f32::type_size());
    TensorHandle::new(
        alloc.memory,
        shape![rows, cols],
        alloc.strides,
        f32::as_type_native_unchecked(),
    )
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
