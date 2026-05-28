//! Matmul as a client of the axis-agnostic [`tile_dsl`] engine: the axis labels
//! `M`, `N`, `K`, the operand roles, the kernel, and the tests. The matmul is one
//! line — `out.mma(&lhs, &rhs)` — over tiles `lhs = {M, K}`, `rhs = {K, N}`,
//! `out = {M, N}`.
#![allow(non_snake_case)]

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, assert_equals_approx};

use super::tile_dsl::{
    Axis, ByAxis, ComputePrimitive, Coverage, CubeDimension, Distribution, Partitioner, Space,
    Spread, Tile, TileKind, TileLaunch, cube_count_for,
};
use super::tile_input::TileInput;

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
    let tile_edge = partitioner.sub_tile_edge(M) as usize;

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    let space = Space::new(&[(M, m), (N, n), (K, k)]);
    let a = TileInput::builder(&client, space.select(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.select(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.select(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .zeros();

    let cube_count = cube_count_for(&partitioner, &space);
    let cube_dim = CubeDim::new_single();

    // The whole matmul is `c.mma(&a, &b)` over the launched tiles.
    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        TileLaunch::new(
            a.view(),
            partitioner.launch(),
            a.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            b.view(),
            partitioner.launch(),
            b.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            c.view(),
            partitioner.launch(),
            c.space(),
            TileKind::GmemWhole,
        ),
        dtype,
        vector_size,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    // Inputs are physical-order aranges over their `[grid, grid, tile, tile]`
    // buffers, so the value the kernel reads at logical `(i, j)` is the element's
    // flat physical index. Build the expected matmul in that same physical order.
    let at = |i: usize, j: usize, cols: usize| -> f32 {
        let grid_c = cols / tile_edge;
        let (gi, ti) = (i / tile_edge, i % tile_edge);
        let (gj, tj) = (j / tile_edge, j % tile_edge);
        (((gi * grid_c + gj) * tile_edge + ti) * tile_edge + tj) as f32
    };
    let (grid_m, grid_n) = (m / tile_edge, n / tile_edge);
    let mut expected = vec![0.0f32; m * n];
    for gm in 0..grid_m {
        for gn in 0..grid_n {
            for tm in 0..tile_edge {
                for tn in 0..tile_edge {
                    let (i, j) = (gm * tile_edge + tm, gn * tile_edge + tn);
                    let value = (0..k).map(|kk| at(i, kk, k) * at(kk, j, n)).sum::<f32>();
                    let offset = ((gm * grid_n + gn) * tile_edge + tm) * tile_edge + tn;
                    expected[offset] = value;
                }
            }
        }
    }
    let (_, expected) = TestInput::builder(
        client,
        shape![m / tile_edge, n / tile_edge, tile_edge, tile_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The kernel: every operand is a [`Tile`] (a semantic view + its space +
/// partitioner), so the whole matmul is one line.
#[cube(launch)]
fn launch_staged_matmul<E: Numeric, S: Size>(
    a: Tile<'_, E, S, CoordsDyn>,
    b: Tile<'_, E, S, CoordsDyn>,
    c: Tile<'_, E, S, CoordsDyn>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    c.mma(&a, &b);
}
