//! Matmul as a client of the axis-agnostic tile DSL engine
#![allow(non_snake_case)]

use cubecl::{
    TestRuntime,
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
    zspace::shape,
};
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

    let cube_count = partitioner.cube_count(&space);
    let cube_dim = CubeDim::new_single();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        TileArgLaunch::new(a.tensor_arg(vector_size), a.space(), a.storage()),
        TileArgLaunch::new(rhs.tensor_arg(vector_size), rhs.space(), rhs.storage()),
        TileArgLaunch::new(c.tensor_arg(vector_size), c.space(), c.storage()),
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

    // The coarsest level drives the launch geometry.
    let cube_count = partitioners[0].cube_count(&out);
    let cube_dim = CubeDim::new_single();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        TileArgLaunch::new(lhs.tensor_arg(vector_size), lhs.space(), lhs.storage()),
        TileArgLaunch::new(rhs.tensor_arg(vector_size), rhs.space(), rhs.storage()),
        TileArgLaunch::new(acc.tensor_arg(vector_size), acc.space(), acc.storage()),
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

    let uses_planes = space.axes().any(|axis| {
        matches!(
            partitioner.distribution(axis),
            Distribution::Spatial {
                scope: ComputeScope::Plane,
                ..
            }
        )
    });
    let plane_size = client.properties().hardware.plane_size_max;
    if uses_planes && plane_size != 1 {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "plane spreading needs plane length 1; backend plane_size = {plane_size}"
        )))
        .enforce();
        return;
    }

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
        partitioner.cube_count(&space),
        partitioner.cube_dim(&client, &space),
        TileArgLaunch::new(a.tensor_arg(1), a.space(), a.storage()),
        TileArgLaunch::new(b.tensor_arg(1), b.space(), b.storage()),
        TileArgLaunch::new(c.tensor_arg(1), c.space(), c.storage()),
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
    check_matmul_multilevel(8, 8, 8, l0, l1);
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
    check_matmul_multilevel(8, 8, 8, l0, l1);
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

    check_matmul_multilevel(8, 8, 8, l0, l1);
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
    check_matmul_multilevel(8, 8, 8, l0, l1);
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
fn check_matmul_multilevel(m: usize, n: usize, k: usize, l0: Partitioner, l1: Partitioner) {
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
        l0.cube_count(&space),
        CubeDim::new_single(),
        TileArgLaunch::new(a.tensor_arg(1), a.space(), a.storage()),
        TileArgLaunch::new(b.tensor_arg(1), b.space(), b.storage()),
        TileArgLaunch::new(c.tensor_arg(1), c.space(), c.storage()),
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
        partitioner.cube_count(&space),
        CubeDim::new_single(),
        TileArgLaunch::new(a.tensor_arg(1), a.space(), a.storage()),
        TileArgLaunch::new(b.tensor_arg(1), b.space(), b.storage()),
        TileArgLaunch::new(c.tensor_arg(1), c.space(), c.storage()),
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
    a: &TileArg<'_, E>,
    b: &TileArg<'_, E>,
    c: &TileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}

/// The CPU kernel: the same `c.mma(a, b)`; the partitioner's `Direct` schedule
/// selects the no-staging move. Operands are size-free — vectorization is a launch
/// concern, not threaded through the DSL.
#[cube(launch)]
fn launch_cpu_matmul<E: Numeric>(
    a: &TileArg<'_, E>,
    b: &TileArg<'_, E>,
    c: &TileArg<'_, E>,
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
/// `Payload::Cmma` transit (`cmma::load_with_layout` / `cmma::store`) preserves data.
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
        TileArgLaunch::new(input.tensor_arg(1), input.space(), input.storage()),
        TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
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
    input: &TileArg<'_, E>,
    output: &TileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = input.tile();
    let space = comptime!(a.space.clone());
    let size = comptime!(space.tile_size());

    let smem_in = Shared::<[Vector<E, Const<1>>]>::new_slice(size);
    let mut a_smem = Tile::smem(&smem_in, comptime!(space.clone()));
    a_smem.stage(&a);
    sync_cube();

    let mut frag = Tile::<Vector<E, Const<1>>>::cmma_fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(space.clone()),
    );
    frag.stage(&a_smem);

    let smem_out = Shared::<[Vector<E, Const<1>>]>::new_slice(size);
    let mut c_smem = Tile::smem(&smem_out, comptime!(space.clone()));
    c_smem.stage(&frag);
    sync_cube();

    let mut c = output.tile();
    c.stage(&c_smem);
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
        TileArgLaunch::new(a.tensor_arg(1), a.space(), a.storage()),
        TileArgLaunch::new(b.tensor_arg(1), b.space(), b.storage()),
        TileArgLaunch::new(c.tensor_arg(1), c.space(), c.storage()),
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

/// gmem A,B → smem → cmma A/B fragments; accumulator init from (zeroed) `c`, then
/// `cmma::execute` (`acc = A·B`), stored back through smem to gmem.
#[cube(launch)]
fn cmma_matmul<E: Numeric>(
    a: &TileArg<'_, E>,
    b: &TileArg<'_, E>,
    c: &TileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();

    let a_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(comptime!(a.space.tile_size()));
    let mut a_smem_tile = Tile::smem(&a_smem, comptime!(a.space.clone()));
    a_smem_tile.stage(&a);

    let b_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(comptime!(b.space.tile_size()));
    let mut b_smem_tile = Tile::smem(&b_smem, comptime!(b.space.clone()));
    b_smem_tile.stage(&b);

    let c_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(comptime!(c.space.tile_size()));
    let mut c_smem_tile = Tile::smem(&c_smem, comptime!(c.space.clone()));
    c_smem_tile.stage(&c);
    sync_cube();

    let mut a_frag = Tile::<Vector<E, Const<1>>>::cmma_fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.space.clone()),
    );
    a_frag.stage(&a_smem_tile);

    let mut b_frag = Tile::<Vector<E, Const<1>>>::cmma_fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.space.clone()),
    );
    b_frag.stage(&b_smem_tile);

    let mut acc = Tile::<Vector<E, Const<1>>>::cmma_fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.space.clone()),
    );
    acc.stage(&c_smem_tile);

    acc.contract_cmma(&a_frag, &b_frag);

    c_smem_tile.stage(&acc);
    sync_cube();
    c.stage(&c_smem_tile);
}
