//! Matmul as a client of the axis-agnostic tile DSL engine
#![allow(non_snake_case)]

use cubecl::{
    TestRuntime,
    cmma::{MatrixIdent, MatrixLayout},
    features::TypeUsage,
    ir::ElemType,
    prelude::*,
    zspace::shape,
};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, TileInput, ValidationResult,
    assert_equals_approx,
};

use cubek_tile::*;

use super::references;

// Matmul's axes — the labels this client gives the engine's opaque `Axis`. `B`
// is the leading batch axis; `M`/`N`/`K` are the matrix axes.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B: Axis = Axis(3);
// A broadcast batch carried as two independent axes: `lhs` spans `B0`, `rhs` spans
// `B1`, the output spans both. Each operand simply omits the axis it broadcasts.
const B0: Axis = Axis(4);
const B1: Axis = Axis(5);

#[test]
fn matmul_sequential_single_cube() {
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
        )
        .staged(),
    );
}

#[test]
fn matmul_one_tile_per_cube() {
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
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (
                    N,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::Y),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (K, Distribution::Sequential),
            ]),
        )
        .staged(),
    );
}

#[test]
fn matmul_reversed_walk_single_cube() {
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
        )
        .staged(),
    );
}

#[test]
fn matmul_contiguous_m_across_cubes() {
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
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .staged(),
    );
}

#[test]
fn matmul_interleaved_m_across_cubes() {
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
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Interleaved,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .staged(),
    );
}

#[test]
fn matmul_batched_walked() {
    check_matmul_batched(3, 8, 8, 8, 4, 1);
}

#[test]
fn matmul_batched_in_sub_tile() {
    check_matmul_batched(4, 8, 8, 8, 4, 4);
}

#[test]
fn matmul_batched_split() {
    check_matmul_batched(4, 8, 8, 8, 4, 2);
}

#[test]
fn matmul_cpu_sequential() {
    check_matmul_cpu(
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
        )
        .direct(),
    );
}

#[test]
fn matmul_cpu_big_k() {
    check_matmul_cpu(
        8,
        8,
        16,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (M, Distribution::Sequential),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .direct(),
    );
}

/// The "global matmul" shape: M and N stay comptime (`Static`), only K is `Dynamic`, so its tile
/// count is resolved from the tensor at runtime while M/N fold and unroll. Exercises the mixed
/// `Static`/`Dynamic` path through `merged_space`/`Extents` that every `all_dynamic` caller skips.
/// Geometry and allocation use the concrete space; the kernel keys on the K-dynamic one.
#[test]
fn matmul_cpu_dynamic_k() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k, edge) = (8usize, 8usize, 16usize, 4usize);
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, edge), (N, edge), (K, edge)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .direct();
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[edge, edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[edge, edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[edge, edge])
        .zeros();

    let dtype = f32::as_type_native_unchecked().storage_type();
    launch_cpu_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        StridedTileArgLaunch::strided(
            a.tensor_arg(1),
            1,
            a.space().with_dynamic(&[K]),
            a.storage(),
        ),
        StridedTileArgLaunch::strided(
            b.tensor_arg(1),
            1,
            b.space().with_dynamic(&[K]),
            b.storage(),
        ),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::tiled_matmul(m, n, k, edge);
    let (_, expected) = TestInput::builder(client, shape![m / edge, n / edge, edge, edge])
        .custom(expected)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

#[test]
fn matmul_cpu_cores_split_m() {
    check_matmul_cpu(
        16,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .direct(),
    );
}

#[test]
fn matmul_cpu_cores_split_m_planes() {
    check_matmul_cpu(
        16,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Plane,
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .direct(),
    );
}

/// Selective batch broadcast over two batch axes `B0 = b0`, `B1 = b1`: `lhs` carries
/// `B0` (and broadcasts `B1`), `rhs` carries `B1` (and broadcasts `B0`). The merge
/// rebuilds the full `{B0, B1}` output batch so every operand reads the right slice.
#[test]
fn matmul_broadcast_two_batch_axes() {
    check_matmul_broadcast(
        4,
        3,
        4,
        &[{
            Partitioner::row_major(
                ByAxis::new(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)]),
                ByAxis::new(&[
                    (B0, Distribution::Sequential),
                    (B1, Distribution::Sequential),
                    (M, Distribution::Sequential),
                    (N, Distribution::Sequential),
                    (K, Distribution::Sequential),
                ]),
            )
            .staged()
        }],
    );
}

#[test]
fn matmul_broadcast_lhs_only() {
    // rhs broadcasts nothing (b0 = 1 makes B0 degenerate); lhs still broadcasts B1.
    check_matmul_broadcast(
        1,
        5,
        4,
        &[{
            let edge = 4;
            Partitioner::row_major(
                ByAxis::new(&[(B0, 1), (B1, 1), (M, edge), (N, edge), (K, edge)]),
                ByAxis::new(&[
                    (B0, Distribution::Sequential),
                    (B1, Distribution::Sequential),
                    (M, Distribution::Sequential),
                    (N, Distribution::Sequential),
                    (K, Distribution::Sequential),
                ]),
            )
            .staged()
        }],
    );
}

/// Both batch axes ride cube-Z at once: `B0` and `B1` are `Spatial { Cube(Z) }`, so
/// the launch puts their *product* on Z and the walk decodes one cube's `CUBE_POS_Z`
/// back into `(b0, b1)`. The same broadcast result as the sequential variants — this
/// is what lets CpuGemm parallelise the whole batch on Z.
#[test]
fn matmul_broadcast_two_batch_axes_on_z() {
    let z = || Distribution::Spatial {
        scope: ComputeScope::Cube(CubeAxis::Z),
        spread: Spread::Contiguous,
        coverage: Coverage::TilesEach(1),
    };
    check_matmul_broadcast(
        4,
        3,
        4,
        &[{
            Partitioner::row_major(
                ByAxis::new(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)]),
                ByAxis::new(&[
                    (B0, z()),
                    (B1, z()),
                    (M, Distribution::Sequential),
                    (N, Distribution::Sequential),
                    (K, Distribution::Sequential),
                ]),
            )
            .staged()
        }],
    );
}

/// The two-axis broadcast tiled across *two* levels: L0 walks the batch
/// (`batch_edge = 1`) and stages the whole `4×4` matrix, then L1 tiles that matrix
/// into `2×2` final tiles. The broadcast (omitted) batch axes must stay correct
/// through both `divide`s. The result is the same broadcast matmul.
#[test]
fn matmul_broadcast_multilevel() {
    check_matmul_broadcast(
        4,
        3,
        4,
        &[
            {
                Partitioner::row_major(
                    ByAxis::new(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)]),
                    ByAxis::new(&[
                        (B0, Distribution::Sequential),
                        (B1, Distribution::Sequential),
                        (M, Distribution::Sequential),
                        (N, Distribution::Sequential),
                        (K, Distribution::Sequential),
                    ]),
                )
                .staged()
            },
            {
                Partitioner::row_major(
                    ByAxis::new(&[(B0, 1), (B1, 1), (M, 2), (N, 2), (K, 2)]),
                    ByAxis::new(&[
                        (B0, Distribution::Sequential),
                        (B1, Distribution::Sequential),
                        (M, Distribution::Sequential),
                        (N, Distribution::Sequential),
                        (K, Distribution::Sequential),
                    ]),
                )
                .staged()
            },
        ],
    );
}

fn check_matmul_batched(
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_edge: usize,
    batch_edge: usize,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    let partitioner = Partitioner::row_major(
        ByAxis::new(&[
            (B, batch_edge),
            (M, tile_edge),
            (N, tile_edge),
            (K, tile_edge),
        ]),
        ByAxis::new(&[
            (B, Distribution::Sequential),
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();

    let space = Space::new(&[(B, b), (M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());
    let a = TileInput::builder(&client, space.project(&[B, M, K]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let rhs = TileInput::builder(&client, space.project(&[B, K, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[B, M, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .zeros();

    let cube_count = space.cube_count();
    let cube_dim = CubeDim::new_single();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        StridedTileArgLaunch::strided(a.tensor_arg(1), vector_size, a.space(), a.storage()),
        StridedTileArgLaunch::strided(rhs.tensor_arg(1), vector_size, rhs.space(), rhs.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), vector_size, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    let expected = references::batched_tiled_matmul(b, m, n, k, tile_edge, batch_edge);
    let (grid_m, grid_n) = (m / tile_edge, n / tile_edge);
    let (_, expected) = TestInput::builder(
        client,
        shape![
            b / batch_edge,
            grid_m,
            grid_n,
            batch_edge,
            tile_edge,
            tile_edge
        ],
    )
    .custom(expected)
    .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// `C = A @ B` where the batch is two independent axes `B0`, `B1` and each operand
/// carries only one: `lhs ∈ {B0, M, K}`, `rhs ∈ {B1, K, N}`, `out ∈ {B0, B1, M, N}`.
/// Each operand omits the batch axis it broadcasts, and the kernel's `Space::merge`
/// fills the omitted axis back wholesale. Single tile per matrix (`t³`) with
/// `batch_edge = 1`, so each output batch element is its own walk point.
fn check_matmul_broadcast(b0: usize, b1: usize, t: usize, partitioners: &[Partitioner]) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    // The one operation space: both batch axes plus a single `t×t` matrix per axis,
    // with the (one or more) partitioner levels attached coarse→fine.
    let space = partitioners.iter().fold(
        Space::new(&[(B0, b0), (B1, b1), (M, t), (N, t), (K, t)]),
        |s, p| s.with_partitioner(p.clone()),
    );

    // Every operand projects onto the shared space; an operand that omits a batch
    // axis broadcasts along all of it (the kernel's `Space::merge` fills it back).
    let out = space.project(&[B0, B1, M, N]);
    let lhs = TileInput::builder(&client, space.project(&[B0, M, K]))
        .tile(&[1, t, t])
        .arange();
    let rhs = TileInput::builder(&client, space.project(&[B1, K, N]))
        .tile(&[1, t, t])
        .arange();
    let acc = TileInput::builder(&client, out.clone())
        .tile(&[1, 1, t, t])
        .zeros();

    // The launch geometry comes off the (whole-tree) space.
    let cube_count = out.cube_count();
    let cube_dim = CubeDim::new_single();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        StridedTileArgLaunch::strided(lhs.tensor_arg(1), vector_size, lhs.space(), lhs.storage()),
        StridedTileArgLaunch::strided(rhs.tensor_arg(1), vector_size, rhs.space(), rhs.storage()),
        StridedTileArgLaunch::strided(acc.tensor_arg(1), vector_size, acc.space(), acc.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, acc.handle(), HostDataType::F32);

    let expected = references::broadcast_matmul(b0, b1, t);
    let (_, expected) = TestInput::builder(client, shape![b0, b1, 1, 1, 1, 1, t, t])
        .custom(expected)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

fn check_matmul_cpu(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    // The CPU register lowering wants the no-staging `Direct` schedule — each caller
    // finalizes its partitioner with `.direct()`.
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());

    let tile_edge = partitioner.edge(M);
    let dtype = f32::as_type_native_unchecked().storage_type();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .zeros();

    launch_cpu_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    let expected = references::tiled_matmul(m, n, k, tile_edge);
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

/// Two different partitioners stacked for multi-level tiling
#[test]
fn matmul_multilevel_staged_then_direct() {
    // Level 0: 4×4×4 blocks, row-major, staged into shared memory.
    let l0 = Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();
    // Level 1: 2×2×2 final tiles within each block, reversed walk, no staging
    let l1 = Partitioner::reversed(
        ByAxis::new(&[(M, 2), (N, 2), (K, 2)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .direct();
    check_matmul_multilevel(8, 8, 8, l0, l1, StageStorage::Strided);
}

#[test]
fn matmul_multilevel_staged_then_staged() {
    let l0 = Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();
    let l1 = Partitioner::row_major(
        ByAxis::new(&[(M, 2), (N, 2), (K, 2)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();
    check_matmul_multilevel(8, 8, 8, l0, l1, StageStorage::Strided);
}

/// Double buffering at the higher level
#[test]
fn matmul_multilevel_double_then_direct() {
    let l0 = Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .double_buffered();

    let l1 = Partitioner::row_major(
        ByAxis::new(&[(M, 2), (N, 2), (K, 2)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .direct();

    check_matmul_multilevel(8, 8, 8, l0, l1, StageStorage::Strided);
}

/// Double buffering at the lower level
#[test]
fn matmul_multilevel_staged_then_double() {
    let seq = || {
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ])
    };
    let l0 = Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq()).staged();
    let l1 =
        Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq()).double_buffered();
    check_matmul_multilevel(8, 8, 8, l0, l1, StageStorage::Strided);
}

/// A storage-tiled stage on a register leaf: the stage layout knob off its default,
/// on any backend (each 4×4 stage cut into contiguous 2×2 blocks).
#[test]
fn matmul_multilevel_tiled_stage() {
    let seq = || {
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ])
    };
    let l0 = Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq()).staged();
    let l1 = Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq()).direct();
    check_matmul_multilevel(8, 8, 8, l0, l1, StageStorage::Tiled);
}

/// A staged level whose walk leaves the lhs unchanged (an N-only walk at L1): the
/// invariant operand fills its slot once, above the loop.
#[test]
fn matmul_staged_invariant_lhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (8usize, 8usize, 8usize);
    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axis(M, seq(4)).axis(N, seq(4)).axis(K, seq(4))
        })
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axis(M, seq(4)).axis(N, seq(2)).axis(K, seq(4))
        })
        .leaf(Leaf::Register);

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The legacy register budget as a level structure: a Direct contraction-step walk
/// (windowing only), a `Staged` N-walk refilling one B fragment per step while the A
/// column fills once above it, and an M-only fragment walk below. Exercises sub-block
/// partition selection (the N-walk's regions each own a column of the accumulator) and
/// the correctness-driven staged unroll. Tensor-core only.
#[test]
fn cmma_matmul_staged_n_walk_partition() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no cmma (tensor-core) support".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages.
        .level(WalkOrder::RowMajor, Schedule::DoubleBuffered, |l| {
            l.axis(M, seq(m)).axis(N, seq(n)).axis(K, seq(stage_k))
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::plane(part))
                .axis(N, Cut::plane(part))
                .axis(K, seq(stage_k))
        })
        // L2: the contraction-step walk, windowing only.
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, seq(part)).axis(N, seq(part)).axis(K, seq(i))
        })
        // L3: the N-walk: one B fragment per step, the A column filled once above it.
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axis(M, seq(part)).axis(N, seq(i)).axis(K, seq(i))
        })
        // L4: the M-only fragment walk.
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, seq(i)).axis(N, seq(i)).axis(K, seq(i))
        })
        .leaf(Leaf::Cmma { k: i });

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

#[test]
fn matmul_double_buffered() {
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
        )
        .double_buffered(),
    );
}

/// Drives the staged lowering with a two-level partitioner stack `[l0, l1]`. `l1`'s
/// edge sizes the final tile (and the data tiling); the coarse `l0` drives launch geometry.
/// `stage` is the operands' stage-layout knob (the output is never staged).
fn check_matmul_multilevel(
    m: usize,
    n: usize,
    k: usize,
    l0: Partitioner,
    l1: Partitioner,
    stage: StageStorage,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let final_edge = l1.edge(M);
    let dtype = f32::as_type_native_unchecked().storage_type();
    let space = Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(l0.clone())
        .with_partitioner(l1.clone());

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[final_edge, final_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[final_edge, final_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[final_edge, final_edge])
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()).stage(stage),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()).stage(stage),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    let expected = references::tiled_matmul(m, n, k, final_edge);
    let (_, expected) = TestInput::builder(
        client,
        shape![m / final_edge, n / final_edge, final_edge, final_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Drives the staged lowering `launch_staged_matmul` for `C = A @ B`.
fn check_matmul(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tile_edge = partitioner.edge(M);
    let dtype = f32::as_type_native_unchecked().storage_type();
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    let expected = references::tiled_matmul(m, n, k, tile_edge);
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

/// The kernel: `c.mma(a, b)` — `c` is a whole tensor, so it lowers; the move comes
/// from its partitioner's `Schedule` (here `.staged()` or `.double_buffered()`).
#[cube(launch)]
fn launch_staged_matmul<E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, E>,
    c: &StridedTileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}

/// The tensor-core kernel: promote the accumulator to its register form (the classic
/// `init_accumulator`), run the whole contraction on it, copy it back (the epilogue).
#[cube(launch)]
fn launch_resident_matmul<E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, E>,
    c: &StridedTileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    let mut acc = c.promote();
    acc.mma(&a, &b);
    c.copy_from(&acc);
}

/// The CPU kernel: the same `c.mma(a, b)`; the partitioner's `Direct` schedule
/// selects the no-staging move. Operands are size-free — vectorization is a launch
/// concern, not threaded through the DSL.
#[cube(launch)]
fn launch_cpu_matmul<E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, E>,
    c: &StridedTileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}

// ---- cmma fragment transit (tensor-core) -------------------------------------

/// Round-trips a 16×16 tile through a tensor-core *accumulator* fragment with no
/// arithmetic: gmem → smem → cmma (load) → smem → gmem (store). Validates that the
/// `TileKind::Cmma` transit (`cmma::load_with_layout` / `cmma::store`) preserves data.
/// Tensor-core only — skipped on backends without cmma (wgpu/cpu); run with
/// `cargo test-metal`.
#[test]
fn cmma_fragment_roundtrip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no cmma (tensor-core) support".to_string(),
        ))
        .enforce();
        return;
    }

    let dtype = f32::as_type_native_unchecked().storage_type();
    let space = Space::new(&[(M, 8), (N, 8)]);

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    cmma_roundtrip::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        StridedTileArgLaunch::strided(input.tensor_arg(1), 1, input.space(), input.storage()),
        StridedTileArgLaunch::strided(output.tensor_arg(1), 1, output.space(), output.storage()),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let want = HostData::from_tensor_handle(&client, input.handle(), HostDataType::F32);
    assert_equals_approx(&got, &want, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// gmem → smem → cmma accumulator → smem → gmem — pure transit, no arithmetic.
#[cube(launch)]
fn cmma_roundtrip<E: Numeric>(
    input: &StridedTileArg<'_, E>,
    output: &StridedTileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = input.tile();
    let space = comptime!(a.space.clone());

    let mut a_smem = MemData::smem(
        comptime!(space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&space)),
    );
    a_smem.copy_from(&a);
    sync_cube();

    let mut frag = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(space.clone()),
    );
    frag.copy_from(&a_smem);

    let mut c_smem = MemData::smem(
        comptime!(space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&space)),
    );
    c_smem.copy_from(&frag);
    sync_cube();

    let mut c = output.tile();
    c.copy_from(&c_smem);
}

/// A real 8×8×8 matmul through tensor cores: `C = A · B`, contracted by `cmma::execute`
/// on the cmma final space. Validates the fragment load → `execute` → store path against
/// the register reference. Tensor-core only — run with `cargo test-metal`.
#[test]
fn cmma_matmul_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no cmma (tensor-core) support".to_string(),
        ))
        .enforce();
        return;
    }

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, Space::new(&[(M, 8), (K, 8)]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();

    cmma_matmul::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::tiled_matmul(8, 8, 8, 8);
    let (_, expected) = TestInput::builder(client, shape![8, 8])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Per-tensor-quantized `A` (i8) through the cmma matmul: `A` dequantizes into smem, then the
/// tensor-core matmul runs in f32. `C = (A·scale)·B`. Needs both cmma and native i8.
#[test]
fn cmma_matmul_quant_per_tensor_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped("backend has no cmma".to_string()))
            .enforce();
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let scale = 0.05f32;
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    // A: i8 quantized, with host values to build the reference.
    let a_dtype = StorageType::Scalar(ElemType::from_quant_value(scheme.value));
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![8, 8])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1, 1])
        .custom(vec![scale])
        .generate_without_host_data();

    // B: f32 row-major arange (b[p, j] = p·8 + j); C: zeros.
    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();

    let a_space = Space::new(&[(M, 8), (K, 8)]);
    let a_storage = Storage::of(2, a_space.rank());
    let e_dtype = f32::as_type_native_unchecked().storage_type();

    cmma_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        StridedTileArgLaunch::strided(a_input.binding().into_tensor_arg(), 1, a_space, a_storage)
            .quantized(scales.binding().into_tensor_arg(), scheme),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale) · (p·8 + j).
    let expected: Vec<f32> = (0..8 * 8)
        .map(|idx| {
            let (i, j) = (idx / 8, idx % 8);
            (0..8)
                .map(|p| (a_host.get_f32(&[i, p]) * scale) * ((p * 8 + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![8, 8])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A matmul through tensor cores with a K walk: the kernel promotes the accumulator to
/// its register-resident form, the staged K regions accumulate into it, and the copy
/// back to gmem is the epilogue. Tensor-core only — run with `cargo test-metal`.
#[test]
fn cmma_matmul_staged_k_walk() {
    check_cmma_matmul_k_walk(16, Schedule::Staged);
}

/// The double-buffered variant: four K regions rotating through two smem slots, the
/// accumulator fragment resident across all of them.
#[test]
fn cmma_matmul_double_buffered_k_walk() {
    check_cmma_matmul_k_walk(32, Schedule::DoubleBuffered);
}

/// An odd region total (three K stages): the loop leaves the last region primed in slot 0;
/// the epilogue must publish and consume it.
#[test]
fn cmma_matmul_double_buffered_odd_k_walk() {
    check_cmma_matmul_k_walk(24, Schedule::DoubleBuffered);
}

/// The K walk staged into a plain strided stage (the legacy `sync_full_strided` storage):
/// the cmma window transport reads through the layout stack either way.
#[test]
fn cmma_matmul_staged_k_walk_strided_stage() {
    check_cmma_matmul_k_walk_v(16, Schedule::Staged, 1, StageStorage::Strided);
}

fn check_cmma_matmul_k_walk(k: usize, schedule: Schedule) {
    check_cmma_matmul_k_walk_v(k, schedule, 1, StageStorage::Tiled)
}

fn check_cmma_matmul_k_walk_v(k: usize, schedule: Schedule, v: usize, stage: StageStorage) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no cmma (tensor-core) support".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, edge) = (8usize, 8usize, 8usize);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, schedule, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .leaf(Leaf::Cmma { k: edge });

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        StridedTileArgLaunch::strided(a.tensor_arg(1), v, a.space(), a.storage()).stage(stage),
        StridedTileArgLaunch::strided(b.tensor_arg(1), v, b.space(), b.storage()).stage(stage),
        StridedTileArgLaunch::strided(c.tensor_arg(1), v, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The multi-plane cmma stage: a double-buffered K walk fills a shared `16×8`/`8×16`
/// stage cooperatively (cyclic across the cube's 128 units), and a plane-partitioned
/// inner level hands each of the 4 planes its own `8×8` fragment, resident across all
/// four K steps. Tensor-core only — run with `cargo test-metal`.
#[test]
fn cmma_matmul_plane_partitioned_stage() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no cmma (tensor-core) support".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, edge) = (16usize, 16usize, 32usize, 8usize);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        // L0: the whole `16×16` output per cube, K walked in `8`-deep stages, double-buffered.
        .level(WalkOrder::RowMajor, Schedule::DoubleBuffered, |l| {
            l.axis(M, Cut::sequential(m))
                .axis(N, Cut::sequential(n))
                .axis(K, Cut::sequential(edge))
        })
        // L1: the stage split one `8×8` fragment per plane.
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::plane(edge))
                .axis(N, Cut::plane(edge))
                .axis(K, Cut::sequential(edge))
        })
        .leaf(Leaf::Cmma { k: edge });

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The multi-fragment partition: each of the 4 planes owns a 2×2 partition of 8³
/// fragments, resident across a double-buffered K walk; the fragment level declares
/// `Direct`, so the static walk reloads operand fragments per execute (no staging).
/// Tensor-core only; run with `cargo test-metal`.
#[test]
fn cmma_matmul_multi_fragment_partition() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no cmma (tensor-core) support".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages.
        .level(WalkOrder::RowMajor, Schedule::DoubleBuffered, |l| {
            l.axis(M, seq(m)).axis(N, seq(n)).axis(K, seq(stage_k))
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::plane(part))
                .axis(N, Cut::plane(part))
                .axis(K, seq(stage_k))
        })
        // L2: the partition level — 2×2 fragments per plane, 2 K sub-tiles.
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, seq(i)).axis(N, seq(i)).axis(K, seq(i))
        })
        .leaf(Leaf::Cmma { k: i });

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// gmem A,B → smem → cmma A/B fragments; accumulator init from (zeroed) `c`, then
/// `cmma::execute` (`acc = A·B`), stored back through smem to gmem.
#[cube(launch)]
fn cmma_matmul<E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, E>,
    c: &StridedTileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();

    let mut a_smem_tile = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&a.space)),
    );
    a_smem_tile.copy_from(&a);

    let mut b_smem_tile = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&b.space)),
    );
    b_smem_tile.copy_from(&b);

    let mut c_smem_tile = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&c.space)),
    );
    c_smem_tile.copy_from(&c);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.space.clone()),
    );
    a_frag.copy_from(&a_smem_tile);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.space.clone()),
    );
    b_frag.copy_from(&b_smem_tile);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.space.clone()),
    );
    acc.copy_from(&c_smem_tile);

    acc.mma(&a_frag, &b_frag);

    c_smem_tile.copy_from(&acc);
    sync_cube();
    c.copy_from(&c_smem_tile);
}

/// Quantized `A`: gmem `I` (i8) dequantized into smem via the `dequantize` op (which threads
/// the storage type `I`) instead of `copy_from` (which can't); `B`/`C` plain `E`. The cmma
/// path then runs entirely in `E`. Mirrors [`cmma_matmul`] otherwise.
#[cube(launch)]
fn cmma_matmul_quant<I: Numeric, E: Numeric>(
    a: &StridedTileArg<'_, I>,
    b: &StridedTileArg<'_, E>,
    c: &StridedTileArg<'_, E>,
    #[define(I)] _idtype: StorageType,
    #[define(E)] _edtype: StorageType,
) {
    let a = a.tile_dequant::<E>();
    let b = b.tile();
    let mut c = c.tile();

    let mut a_smem = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&a.space)),
    );
    a_smem.dequantize_from::<I>(&a);

    let mut b_smem = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&b.space)),
    );
    b_smem.copy_from(&b);

    let mut c_smem = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        comptime!(StagePlan::for_space(&c.space)),
    );
    c_smem.copy_from(&c);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.space.clone()),
    );
    a_frag.copy_from(&a_smem);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.space.clone()),
    );
    b_frag.copy_from(&b_smem);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.space.clone()),
    );
    acc.copy_from(&c_smem);

    acc.mma(&a_frag, &b_frag);

    c_smem.copy_from(&acc);
    sync_cube();
    c.copy_from(&c_smem);
}

/// Block-quantized `A` (block along `M`): `A`'s space tiles `M` into `bm`-deep blocks so the
/// smem dequant fill descends to one scale per block; the cmma fragment still reads the whole
/// `8×8` smem. Validates block windowing survives into the matmul stage.
#[test]
fn cmma_matmul_quant_block_m_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped("backend has no cmma".to_string()))
            .enforce();
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let bm = 4usize; // 2 blocks along M, each 4×8; one scale each
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([bm as u8, 8]))
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    let a_dtype = StorageType::Scalar(ElemType::from_quant_value(scheme.value));
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![8, 8])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    // One distinct scale per M-block: scales shaped (8/bm, 1).
    let scale_vals: Vec<f32> = (0..8 / bm).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), shape![8 / bm, 1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    // A's space tiles M into `bm`-blocks so the dequant fill descends to one scale per block.
    let a_space = Tiling::new()
        .extents(&[(M, 8), (K, 8)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(bm)).axis(K, Cut::sequential(8))
        })
        .leaf(Leaf::Register);
    let a_storage = Storage::of(2, a_space.rank());

    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();
    let e_dtype = f32::as_type_native_unchecked().storage_type();

    cmma_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        StridedTileArgLaunch::strided(a_input.binding().into_tensor_arg(), 1, a_space, a_storage)
            .quantized(scales.binding().into_tensor_arg(), scheme),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale[i/bm]) · (p·8 + j).
    let expected: Vec<f32> = (0..8 * 8)
        .map(|idx| {
            let (i, j) = (idx / 8, idx % 8);
            let scale = scale_vals[i / bm];
            (0..8)
                .map(|p| (a_host.get_f32(&[i, p]) * scale) * ((p * 8 + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![8, 8])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Block-quantized `A` along `K` (the contraction axis): the scale changes partway through
/// each dot product. `A`'s space tiles `K` into `bk`-deep blocks so the smem dequant picks the
/// right scale per K-block. The case that matters for quantized-weight matmul.
#[test]
fn cmma_matmul_quant_block_k_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped("backend has no cmma".to_string()))
            .enforce();
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let bk = 4usize; // 2 blocks along K, each 8×4; the scale changes at p = 4
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([8, bk as u8]))
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    let a_dtype = StorageType::Scalar(ElemType::from_quant_value(scheme.value));
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![8, 8])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    // One distinct scale per K-block: scales shaped (1, 8/bk).
    let scale_vals: Vec<f32> = (0..8 / bk).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), shape![1, 8 / bk])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    // A's space tiles K into `bk`-blocks so the dequant fill picks a scale per K-block.
    let a_space = Tiling::new()
        .extents(&[(M, 8), (K, 8)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(8)).axis(K, Cut::sequential(bk))
        })
        .leaf(Leaf::Register);
    let a_storage = Storage::of(2, a_space.rank());

    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();
    let e_dtype = f32::as_type_native_unchecked().storage_type();

    cmma_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        StridedTileArgLaunch::strided(a_input.binding().into_tensor_arg(), 1, a_space, a_storage)
            .quantized(scales.binding().into_tensor_arg(), scheme),
        StridedTileArgLaunch::strided(b.tensor_arg(1), 1, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale[p/bk]) · (p·8 + j).
    let expected: Vec<f32> = (0..8 * 8)
        .map(|idx| {
            let (i, j) = (idx / 8, idx % 8);
            (0..8)
                .map(|p| (a_host.get_f32(&[i, p]) * scale_vals[p / bk]) * ((p * 8 + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![8, 8])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Vectorized operands (2-wide lines) through the Direct schedule: gmem-only line-unit
/// addressing. Regression for the line-vs-scalar unit bug (worked on cubecl-cpu only).
#[test]
fn matmul_direct_vectorized() {
    check_matmul_vectorized(Schedule::Direct);
}

/// Vectorized operands through the staged schedule: the cooperative fill moves lines
/// through smem. Regression for the line-vs-scalar unit bug.
#[test]
fn matmul_staged_vectorized() {
    check_matmul_vectorized(Schedule::Staged);
}

fn check_matmul_vectorized(schedule: Schedule) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k, edge, v) = (8usize, 8usize, 8usize, 4usize, 2usize);
    let builder = Partitioner::row_major(
        ByAxis::new(&[(M, edge), (N, edge), (K, edge)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    );
    let partitioner = match schedule {
        Schedule::Direct => builder.direct(),
        Schedule::Staged => builder.staged(),
        Schedule::DoubleBuffered => builder.double_buffered(),
    };
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        StridedTileArgLaunch::strided(a.tensor_arg(1), v, a.space(), a.storage()),
        StridedTileArgLaunch::strided(b.tensor_arg(1), v, b.space(), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), v, c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The staged cmma K walk with operands served in 2-wide lines: the cooperative fill
/// moves lines, the cmma transport addresses the scalar buffer underneath.
#[test]
fn cmma_matmul_staged_k_walk_vectorized() {
    check_cmma_matmul_k_walk_v(16, Schedule::Staged, 2, StageStorage::Tiled);
}
