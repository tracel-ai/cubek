//! Matmul as a client of the axis-agnostic [`tile_dsl`] engine: the axis labels
//! `M`, `N`, `K`, the operand roles, the kernel, and the tests. The matmul is one
//! line — `out.mma(&lhs, &rhs)` — over tiles `lhs = {M, K}`, `rhs = {K, N}`,
//! `out = {M, N}`.
#![allow(non_snake_case)]

use cubecl::std::tensor::layout::{CoordsDyn, tiled_view::tiled_view};
use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{
    HostData, HostDataType, LayoutSpec, StridedLayout, TestInput, assert_equals_approx,
};

use super::tile_dsl::{
    Axis, ByAxis, ComputePrimitive, Coverage, CubeDimension, Distribution, Partitioner, Space,
    Spread, Tile, TileKind, TileLaunch, cube_count_for,
};

// Matmul's three axes — the labels this client gives the engine's opaque `Axis`.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Staged matmul on tile-permuted tensors, single cube: every axis is
/// `Sequential`, so the `RowMajor` partitioner walks every output tile in turn.
#[test]
fn ___matmul_sequential_single_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (M, Distribution::Sequential),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// One tile per cube: M and N are pinned to 2 cube instances each; with
/// `grid = 2` on both, that's one output tile per cube while K stays sequential.
#[test]
fn ___matmul_one_tile_per_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (
                    N,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::Y),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// The same single-cube matmul, `Reversed`: visits output tiles back-to-front,
/// same result.
#[test]
fn ___matmul_reversed_walk_single_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::reversed(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (M, Distribution::Sequential),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// Contiguous spread sized by `TilesEach`: each cube does 2 contiguous m-tiles,
/// instance count derived (`grid_m / 2 = 2` cubes).
#[test]
fn ___matmul_contiguous_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// Interleaved spread sized by `Instances`: M split across 2 cubes round-robin
/// (cube 0 → `{0,2}`, cube 1 → `{1,3}`).
#[test]
fn ___matmul_interleaved_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::X),
                        spread: Spread::Interleaved,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// Drives `launch_staged_matmul` for `C = A @ B` under an arbitrary
/// `partitioner`; the launch geometry is derived from it via [`cube_count_for`].
fn check_matmul(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Square tiles here, so M's sub-tile edge is the tile size for the layout.
    let tile_size = partitioner.sub_tile_edge(M) as usize;
    let shape = shape![m, n];
    let shape_a = shape![m, k];
    let shape_b = shape![k, n];

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    let tiled_layout = LayoutSpec::tiled(
        StridedLayout::RowMajor,
        0,
        vec![tile_size as u16, tile_size as u16],
    );

    let a_storage = build_tile_permuted(m, k, tile_size, |i, j| (i * k + j) as f32);
    let b_storage = build_tile_permuted(k, n, tile_size, |i, j| (i * n + j) as f32);

    let a_handle = TestInput::builder(client.clone(), shape_a.clone())
        .layout(tiled_layout.clone())
        .custom(a_storage)
        .generate();
    let b_handle = TestInput::builder(client.clone(), shape_b.clone())
        .layout(tiled_layout.clone())
        .custom(b_storage)
        .generate();
    let c_handle = TestInput::builder(client.clone(), shape.clone())
        .layout(tiled_layout.clone())
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, m as u32), (N, n as u32), (K, k as u32)]);
    let cube_count = cube_count_for(&partitioner, &space);
    let cube_dim = CubeDim::new_single();

    // Each operand is a launchable Tile: a semantic view + the space it lives in
    // + the partitioner. The kernel just does `c.mma(&a, &b)`.
    let tiles = vec![tile_size as u16, tile_size as u16];
    let a_tile = TileLaunch::new(
        tiled_view(a_handle.binding(), 0, tiles.clone()),
        partitioner.launch(),
        Space::new(&[(M, m as u32), (K, k as u32)]),
        TileKind::GmemWhole,
    );
    let b_tile = TileLaunch::new(
        tiled_view(b_handle.binding(), 0, tiles.clone()),
        partitioner.launch(),
        Space::new(&[(K, k as u32), (N, n as u32)]),
        TileKind::GmemWhole,
    );
    let c_tile = TileLaunch::new(
        tiled_view(c_handle.clone().binding(), 0, tiles),
        partitioner.launch(),
        Space::new(&[(M, m as u32), (N, n as u32)]),
        TileKind::GmemWhole,
    );

    launch_staged_matmul::launch::<TestRuntime>(
        &client, cube_count, cube_dim, a_tile, b_tile, c_tile, dtype, vector_size,
    );

    let output = HostData::from_tensor_handle(&client, c_handle, HostDataType::F32);

    let c_storage = build_tile_permuted(m, n, tile_size, |i, j| {
        (0..k)
            .map(|kk| (i * k + kk) as f32 * (kk * n + j) as f32)
            .sum::<f32>()
    });
    let (_, expected_values) = TestInput::builder(client, shape)
        .layout(tiled_layout)
        .custom(c_storage)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected_values, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Build a buffer whose *storage order* matches a 2D tiled layout
/// `[Grid_m, Grid_n, Tile_m, Tile_n]`, populated so the semantic-coord view
/// returns `semantic_at(i, j)` at position `(i, j)`.
fn build_tile_permuted<F: Fn(usize, usize) -> f32>(
    rows: usize,
    cols: usize,
    tile: usize,
    semantic_at: F,
) -> Vec<f32> {
    let grid_rows = rows / tile;
    let grid_cols = cols / tile;
    let mut out = vec![0.0; rows * cols];
    for gm in 0..grid_rows {
        for gn in 0..grid_cols {
            for tm in 0..tile {
                for tn in 0..tile {
                    let storage_offset =
                        gm * grid_cols * tile * tile + gn * tile * tile + tm * tile + tn;
                    out[storage_offset] = semantic_at(gm * tile + tm, gn * tile + tn);
                }
            }
        }
    }
    out
}

/// The kernel: every operand is a [`Tile`] (a semantic view + its space +
/// partitioner), so the whole matmul is one line.
#[cube(launch)]
fn launch_staged_matmul<E: Numeric, S: Size>(
    a: Tile<E, S, CoordsDyn>,
    b: Tile<E, S, CoordsDyn>,
    c: Tile<E, S, CoordsDyn>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    c.mma(&a, &b);
}
