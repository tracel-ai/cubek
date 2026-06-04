//! Matmul as a client of the axis-agnostic tile DSL engine: the axis labels
//! `M`, `N`, `K`, the operand roles, the kernels, and the tests. Over tiles
//! `lhs = {M, K}`, `rhs = {K, N}`, `out = {M, N}` — the staged matmul is the
//! DSL's built-in [`mma_staged`]; the CPU lowering below reconstructs the matmul
//! space from the operands, partitions it, and projects, matmul-specific (operand
//! roles + scalar contraction) so it lives with the client, not in the DSL.
#![allow(non_snake_case)]

use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, TileInput, ValidationResult,
    assert_equals_approx,
};

// Glob brings the tile-DSL items *and* the cube-macro-generated `*Expand`
// companions the lowering below needs.
use cubek_tile::*;

use super::references;

// Matmul's axes — the labels this client gives the engine's opaque `Axis`. `B`
// is the leading batch axis; `M`/`N`/`K` are the matrix axes.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B: Axis = Axis(3);

/// Staged matmul on tile-permuted tensors, single cube: every axis is
/// `Sequential`, so the `RowMajor` partitioner walks every output tile in turn.
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
        ),
    );
}

/// One tile per cube: M and N are pinned to 2 cube instances each; with
/// `grid = 2` on both, that's one output tile per cube while K stays sequential.
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
        ),
    );
}

/// The same single-cube matmul, `Reversed`: visits output tiles back-to-front,
/// same result.
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
        ),
    );
}

/// Contiguous spread sized by `TilesEach`: each cube does 2 contiguous m-tiles,
/// instance count derived (`grid_m / 2 = 2` cubes).
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
        ),
    );
}

/// Interleaved spread sized by `Instances`: M split across 2 cubes round-robin
/// (cube 0 → `{0,2}`, cube 1 → `{1,3}`).
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
        ),
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

/// Single core: no staging, register microkernel.
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
        ),
    );
}

/// Big K: four K-tiles accumulate into one output block (`grid_k == 1` is the
/// lucky single-write case).
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
        ),
    );
}

/// M split across cores via cubes (sequential iterations on this backend).
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
        ),
    );
}

/// M split across cores via planes (one plane = one core).
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
        ),
    );
}

/// Selective batch broadcast over a *squashed* batch axis `B = [b0, b1]`: `lhs`
/// carries `[b0, 1]` (broadcasts the second batch dim), `rhs` carries `[1, b1]`
/// (broadcasts the first). The composite extent unravels each output batch back
/// into `(g0, g1)` so every operand reads the right slice.
#[test]
fn matmul_broadcast_squashed_batch() {
    check_matmul_broadcast(4, 3, 4, &[broadcast_level(4)]);
}

#[test]
fn matmul_broadcast_squashed_batch_lhs_only() {
    // rhs broadcasts nothing (b0 = 1 on the first dim is degenerate); lhs still
    // broadcasts the trailing batch dim.
    check_matmul_broadcast(1, 5, 4, &[broadcast_level(4)]);
}

/// The composite (squashed-batch) broadcast, now tiled across *two* levels: L0
/// walks the batch (`batch_edge = 1`) and stages the whole `4×4` matrix, then L1
/// tiles that matrix into `2×2` leaves. The batch factorization `[b0, b1]` must
/// survive L0's `divide` (→ `[1, 1]`) for the broadcast to stay correct — the point
/// of the composite-extent fix. The result is the same broadcast matmul.
#[test]
fn matmul_broadcast_squashed_batch_multilevel() {
    check_matmul_broadcast(4, 3, 4, &[broadcast_level(4), broadcast_level(2)]);
}

/// A broadcast partitioner level: batch walked one element at a time (`B` edge 1),
/// matrix axes cut to `edge`. All-sequential, single cube.
fn broadcast_level(edge: usize) -> Partitioner {
    Partitioner::row_major(
        ByAxis::new(&[(B, 1), (M, edge), (N, edge), (K, edge)]),
        ByAxis::new(&[
            (B, Distribution::Sequential),
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
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
    );

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

    let cube_count = cube_count_for(&partitioner, &space);
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
        shape![b / batch_edge, grid_m, grid_n, batch_edge, tile_edge, tile_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// `C = A @ B` where the batch axis is a squashed `B = [b0, b1]` and each operand
/// broadcasts a different batch dimension: `lhs ∈ {B[b0,1], M, K}`, `rhs ∈
/// {B[1,b1], K, N}`, `out ∈ {B[b0,b1], M, N}`. Single tile per matrix (`t³`) so
/// the batch is the only thing being walked, and `batch_edge = 1` so each output
/// batch element is its own walk point — exercising the composite-extent unravel
/// in `project`.
fn check_matmul_broadcast(b0: usize, b1: usize, t: usize, levels: &[Partitioner]) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    // Apply the (one or more) partitioner levels to an operand space, coarse→fine.
    let attach = |space: Space| {
        levels
            .iter()
            .fold(space, |s, p| s.with_partitioner(p.clone()))
    };

    // Each operand declares its own batch factorization; the `1`s are the
    // broadcast dimensions. The kernel's `Space::merge` combines them to `[b0, b1]`.
    let out_space = Space::with_levels(&[
        (B, Extent::new(&[b0, b1])),
        (M, Extent::scalar(t)),
        (N, Extent::scalar(t)),
    ]);
    let lhs_space = Space::with_levels(&[
        (B, Extent::new(&[b0, 1])),
        (M, Extent::scalar(t)),
        (K, Extent::scalar(t)),
    ]);
    let rhs_space = Space::with_levels(&[
        (B, Extent::new(&[1, b1])),
        (K, Extent::scalar(t)),
        (N, Extent::scalar(t)),
    ]);

    let a = TileInput::builder(&client, attach(lhs_space))
        .tile(&[1, t, t])
        .arange();
    let rhs = TileInput::builder(&client, attach(rhs_space))
        .tile(&[1, t, t])
        .arange();
    let c = TileInput::builder(&client, attach(out_space.clone()))
        .tile(&[1, t, t])
        .zeros();

    // The coarsest level drives the launch geometry.
    let cube_count = cube_count_for(&levels[0], &out_space);
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

    let expected = references::broadcast_matmul(b0, b1, t);
    let (_, expected) = TestInput::builder(client, shape![b0 * b1, 1, 1, 1, t, t])
        .custom(expected)
        .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Drives the CPU lowering for `C = A @ B`. The contraction is scalar and the DSL
/// is size-free, so every operand is a plain (`vector_size = 1`) tile in the full
/// element space — no line-unit bookkeeping. Plane spreading models one core per
/// plane (length 1); a wider-plane backend rejects (see `cube_dim_for`), so skip.
fn check_matmul_cpu(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    // The CPU register lowering is the no-staging `Direct` schedule.
    let partitioner = partitioner.direct();
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

    let tile_edge = partitioner.sub_tile_edge(M) as usize;
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
        cube_count_for(&partitioner, &space),
        cube_dim_for(&client, &partitioner, &space),
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

/// Two *different* partitioners stacked coarse→fine, exercising multi-level tiling
/// in the kernel: `c.mma(a, b)` lowers through `l0` (the head level), and each
/// projected sub-tile — still carrying `l1` — re-lowers, so the kernel descends a
/// second walk before reaching the contraction leaf. `l0`/`l1` may differ in every
/// way (edge, walk order, schedule); the result is still the plain matmul. The data
/// is tiled to the *leaf* edge, so the expected layout matches `check_matmul`.
#[test]
fn matmul_multilevel_staged_then_direct() {
    // L0: 4×4×4 blocks, row-major, staged into shared memory.
    let l0 = Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    );
    // L1: 2×2×2 leaf tiles within each block, reversed walk, no staging (Direct) —
    // a totally different level, contracting straight from the staged L0 block.
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

/// Both levels stage: the fine level re-stages each coarse shared-memory block into
/// a *second* shared buffer before contracting (smem→smem). This path only exists
/// if the kernel actually descends into the second level — a single-level collapse
/// could never allocate the inner buffer or copy from an smem source.
#[test]
fn matmul_multilevel_staged_then_staged() {
    let seq = || {
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ])
    };
    let l0 = Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq());
    let l1 = Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq());
    check_matmul_multilevel(8, 8, 8, l0, l1);
}

/// Double buffering at the *coarse* level of a multi-level stack: L0 pipelines its
/// `4×4×4` blocks through the `Ring` (prefetch next while computing current), and
/// each block — still carrying L1 — re-lowers `Direct` into `2×2×2` leaves. The
/// pipeline composes with the inner walk via the same `out.at(region).mma(…)`
/// recursion; the result is the plain matmul.
#[test]
fn matmul_multilevel_double_then_direct() {
    let seq = || {
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ])
    };
    let l0 =
        Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq()).double_buffered();
    let l1 = Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq()).direct();
    check_matmul_multilevel(8, 8, 8, l0, l1);
}

/// Double buffering at the *fine* level: L0 stages each `4×4×4` block into shared
/// memory, then L1 pipelines that block's `2×2×2` leaves through its own `Ring`.
/// Exercises a per-level ring allocated within the outer walk and fed from an smem
/// block.
#[test]
fn matmul_multilevel_staged_then_double() {
    let seq = || {
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ])
    };
    let l0 = Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq());
    let l1 =
        Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq()).double_buffered();
    check_matmul_multilevel(8, 8, 8, l0, l1);
}

/// Drives the staged lowering with a two-level partitioner stack `[l0, l1]`. `l1`'s
/// edge sizes the leaf (and the data tiling); the coarse `l0` drives launch geometry.
fn check_matmul_multilevel(m: usize, n: usize, k: usize, l0: Partitioner, l1: Partitioner) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let leaf_edge = l1.sub_tile_edge(M) as usize;
    let dtype = f32::as_type_native_unchecked().storage_type();
    let space = Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(l0.clone())
        .with_partitioner(l1.clone());

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[leaf_edge, leaf_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[leaf_edge, leaf_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[leaf_edge, leaf_edge])
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count_for(&l0, &space),
        CubeDim::new_single(),
        TileArgLaunch::new(a.tensor_arg(1), a.space(), a.storage()),
        TileArgLaunch::new(b.tensor_arg(1), b.space(), b.storage()),
        TileArgLaunch::new(c.tensor_arg(1), c.space(), c.storage()),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    let expected = references::tiled_matmul(m, n, k, leaf_edge);
    let (_, expected) = TestInput::builder(
        client,
        shape![m / leaf_edge, n / leaf_edge, leaf_edge, leaf_edge],
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
    let tile_edge = partitioner.sub_tile_edge(M) as usize;
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
        cube_count_for(&partitioner, &space),
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
/// from its partitioner's `Schedule` (staged by default, or double-buffered).
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

/// Double-buffered matmul: the same `check_matmul` driver, but the partitioner
/// carries the `DoubleBuffered` schedule, so `c.mma(a, b)` lowers through the
/// `Ring` pipeline (`mma_double` in the DSL).
fn double_sequential() -> Partitioner {
    Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .double_buffered()
}

#[test]
fn matmul_double_buffered() {
    check_matmul(8, 8, 8, double_sequential());
}
