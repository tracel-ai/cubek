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
use cubek_quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, TileInput, ValidationResult,
    assert_equals_approx,
};

use cubek_tile::*;

use super::references;

/// Skip guard for the tensor-core tests in this file, which all hardcode
/// `8x8x8` `f32` fragments (the native Metal simdgroup shape). Checking only
/// that *some* cmma config exists is not enough: drivers accept only the exact
/// fragment shapes they advertise, and an unsupported shape is rejected at
/// compile time. Returns `false` (after enforcing a skip outcome) when the
/// device doesn't advertise the exact configuration.
fn require_cmma_8x8x8_f32(client: &ComputeClient<TestRuntime>) -> bool {
    let f32_ty = f32::elem_type_native();
    let supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f32_ty
            && cfg.b_type == f32_ty
            && cfg.cd_type == f32_ty
            && cfg.m == 8
            && cfg.n == 8
            && cfg.k == 8
    });
    if !supported {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no 8x8x8 f32 cmma (tensor-core) fragment support".to_string(),
        ))
        .enforce();
    }
    supported
}

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

// A second contracted axis, so a contraction that is otherwise a plain matmul takes the N-D nest.
const K2: Axis = Axis(6);

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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
    .buffered(Buffering::SINGLE);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[edge, edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[edge, edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[edge, edge])
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_cpu_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space
            .with_dynamic(&[K])
            .with_instruction(Instruction::registers(16)),
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
        .buffered(Buffering::SINGLE),
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
        .buffered(Buffering::SINGLE),
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
            .buffered(Buffering::SINGLE)
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
            .buffered(Buffering::SINGLE)
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
            .buffered(Buffering::SINGLE)
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
                .buffered(Buffering::SINGLE)
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
                .buffered(Buffering::SINGLE)
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

    let dtype = f32::elem_type_native();
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
    .buffered(Buffering::SINGLE);

    let space = Space::new(&[(B, b), (M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());
    let mut a_operand = Operand::new(&[B, M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    let a = TileInput::builder(&client, space.project(&[B, M, K]))
        .operand(&a_operand)
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let mut rhs_operand = Operand::new(&[B, K, N], f32::elem_type_native());
    rhs_operand.stage(Residence::Smem);
    let rhs = TileInput::builder(&client, space.project(&[B, K, N]))
        .operand(&rhs_operand)
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
        vector_size,
        a.arg(),
        rhs.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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

    let dtype = f32::elem_type_native();
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
    // Every level of this helper stages, whatever the caller stacked.
    let stages = vec![Residence::Smem; partitioners.len()];
    let mut lhs_operand = Operand::new(&[B0, M, K], f32::elem_type_native());
    for &residence in &stages {
        lhs_operand.stage(residence);
    }
    let lhs = TileInput::builder(&client, space.project(&[B0, M, K]))
        .operand(&lhs_operand)
        .tile(&[1, t, t])
        .arange();
    let mut rhs_operand = Operand::new(&[B1, K, N], f32::elem_type_native());
    for &residence in &stages {
        rhs_operand.stage(residence);
    }
    let rhs = TileInput::builder(&client, space.project(&[B1, K, N]))
        .operand(&rhs_operand)
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
        vector_size,
        lhs.arg(),
        rhs.arg(),
        acc.arg(),
        space.with_instruction(Instruction::registers(16)),
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
    // The CPU register lowering reads both operands where they lie: the inputs state no
    // residence, so the level materializes nothing and the walk is the plain recursion.
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());

    let tile_edge = partitioner.edge(M);
    let dtype = f32::elem_type_native();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(4242, 10., 100.);

    launch_cpu_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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
    .buffered(Buffering::SINGLE);
    // Level 1: 2×2×2 final tiles within each block, reversed walk, no staging
    let l1 = Partitioner::reversed(
        ByAxis::new(&[(M, 2), (N, 2), (K, 2)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);
    check_matmul_multilevel(
        8,
        8,
        8,
        l0,
        l1,
        StageStorage::Strided,
        &[Residence::Smem, Residence::InPlace],
    );
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
    .buffered(Buffering::SINGLE);
    let l1 = Partitioner::row_major(
        ByAxis::new(&[(M, 2), (N, 2), (K, 2)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);
    check_matmul_multilevel(
        8,
        8,
        8,
        l0,
        l1,
        StageStorage::Strided,
        &[Residence::Smem, Residence::Smem],
    );
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
    .buffered(Buffering::DOUBLE);

    let l1 = Partitioner::row_major(
        ByAxis::new(&[(M, 2), (N, 2), (K, 2)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);

    check_matmul_multilevel(
        8,
        8,
        8,
        l0,
        l1,
        StageStorage::Strided,
        &[Residence::Smem, Residence::InPlace],
    );
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
    let l0 = Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq())
        .buffered(Buffering::SINGLE);
    let l1 = Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq())
        .buffered(Buffering::DOUBLE);
    check_matmul_multilevel(
        8,
        8,
        8,
        l0,
        l1,
        StageStorage::Strided,
        &[Residence::Smem, Residence::Smem],
    );
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
    let l0 = Partitioner::row_major(ByAxis::new(&[(M, 4), (N, 4), (K, 4)]), seq())
        .buffered(Buffering::SINGLE);
    let l1 = Partitioner::row_major(ByAxis::new(&[(M, 2), (N, 2), (K, 2)]), seq())
        .buffered(Buffering::SINGLE);
    check_matmul_multilevel(
        8,
        8,
        8,
        l0,
        l1,
        StageStorage::Tiled,
        &[Residence::Smem, Residence::InPlace],
    );
}

/// A staged level whose walk leaves the lhs unchanged (an N-only walk at L1): the
/// invariant operand fills its slot once, above the loop.
#[test]
fn matmul_staged_invariant_lhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (8usize, 8usize, 8usize);
    let seq = |edge| Cut::sequential(edge);
    let dtype = f32::elem_type_native();
    let mut ops = (
        Operand::new(&[M, K], dtype),
        Operand::new(&[K, N], dtype),
        Operand::new(&[M, N], dtype),
    );
    let space = Tiling::over(&mut ops, &[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, seq(4)).axis(N, seq(4)).axis(K, seq(4));
            o.0.stage(Residence::Smem);
            o.1.stage(Residence::Smem);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, seq(4)).axis(N, seq(2)).axis(K, seq(4));
            o.0.stage(Residence::Smem);
            o.1.stage(Residence::Smem);
        })
        .build();

    let a = TileInput::builder(&client, space.project(ops.0.axes()))
        .operand(&ops.0)
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(ops.1.axes()))
        .operand(&ops.1)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(ops.2.axes()))
        .untiled()
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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

/// N spread across a plane's lanes (`ComputeScope::Unit`): each lane owns a disjoint
/// column of the register-leaf output and contracts the whole K in registers — the
/// gemv-perpendicular mapping. `Cut::unit` declares the split without the lane count;
/// [`Space::resolve_lanes`] (the launch's stamping pass) fills it from the hardware
/// `plane_size`, so the Unit axis rides the warp's lanes on the cube's X dim.
/// `plane_size == 1` on CPU degenerates to one lane doing all of N (still correct); the
/// win is on GPU where the warp's lanes divide N.
#[test]
fn register_matmul_unit_spread_n() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;

    let (m, k, nr) = (4usize, 8usize, 2usize);
    let n = lanes * nr;
    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, seq(m)).axis(N, Cut::unit(nr)).axis(K, seq(k))
        })
        .build()
        // The launcher's stamping pass: `Cut::unit`'s deferred count becomes `plane_size`.
        .resolve_lanes(lanes);

    let dtype = f32::elem_type_native();
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
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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

/// The legacy register budget as a level structure: an in-place contraction-step walk
/// (windowing only), an N-walk refilling one B fragment per step while the A
/// column fills once above it, and an M-only fragment walk below. Exercises sub-block
/// partition selection (the N-walk's regions each own a column of the accumulator) and
/// the correctness-driven staged unroll. Tensor-core only.
#[test]
fn cmma_matmul_staged_n_walk_partition() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let seq = |edge| Cut::sequential(edge);
    let instruction = Instruction::Cmma;
    let dtype = f32::elem_type_native();
    let mut ops = (
        Operand::new(&[M, K], dtype),
        Operand::new(&[K, N], dtype),
        Operand::new(&[M, N], dtype),
    );
    let space = Tiling::over(&mut ops, &[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages; both
        // inputs take a shared stage there.
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l, o| {
            l.axis(M, seq(m)).axis(N, seq(n)).axis(K, seq(stage_k));
            o.0.stage(Residence::Smem);
            o.1.stage(Residence::Smem);
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::plane(part))
                .axis(N, Cut::plane(part))
                .axis(K, seq(stage_k));
        })
        // L2: the contraction-step walk, windowing only.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, seq(part)).axis(N, seq(part)).axis(K, seq(i));
        })
        // L3: the N-walk: one B fragment per step, the A column filled once above it.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, seq(part)).axis(N, seq(i)).axis(K, seq(i));
            o.0.stage(Residence::Register);
            o.1.stage(Residence::Register);
        })
        // L4: the M-only fragment walk.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, seq(i)).axis(N, seq(i)).axis(K, seq(i));
        })
        .build();

    let a = TileInput::builder(&client, space.project(ops.0.axes()))
        .operand(&ops.0)
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(ops.1.axes()))
        .operand(&ops.1)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(ops.2.axes()))
        .untiled()
        // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
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

/// Double-buffered walk over a plane-partition stage (Residence::Register) under a CMMA leaf.
/// Exercises the unrolled pipelined walk (unroll == true) with register partition selection.
#[test]
fn cmma_matmul_double_buffered_plane_stage() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let seq = |edge| Cut::sequential(edge);
    let instruction = Instruction::Cmma;
    let dtype = f32::elem_type_native();
    let mut ops = (
        Operand::new(&[M, K], dtype),
        Operand::new(&[K, N], dtype),
        Operand::new(&[M, N], dtype),
    );
    let space = Tiling::over(&mut ops, &[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages; both
        // inputs take a shared stage there.
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l, o| {
            l.axis(M, seq(m)).axis(N, seq(n)).axis(K, seq(stage_k));
            o.0.stage(Residence::Smem);
            o.1.stage(Residence::Smem);
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::plane(part))
                .axis(N, Cut::plane(part))
                .axis(K, seq(stage_k));
        })
        // L2: the contraction-step walk, windowing only.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, seq(part)).axis(N, seq(part)).axis(K, seq(i));
        })
        // L3: the N-walk with DOUBLE buffering over a plane stage.
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l, o| {
            l.axis(M, seq(part)).axis(N, seq(i)).axis(K, seq(i));
            o.0.stage(Residence::Register);
            o.1.stage(Residence::Register);
        })
        // L4: the M-only fragment walk.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, seq(i)).axis(N, seq(i)).axis(K, seq(i));
        })
        .build();

    let a = TileInput::builder(&client, space.project(ops.0.axes()))
        .operand(&ops.0)
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(ops.1.axes()))
        .operand(&ops.1)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(ops.2.axes()))
        .untiled()
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
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
        .buffered(Buffering::DOUBLE),
    );
}

/// Double buffering with only the lhs staged: `a` takes a shared stage while `b` is read straight
/// from global memory, in one slot, on a level that prefetches. How deep a level buffers and where
/// each of its operands lives are independent, and this is the pair that could not be expressed
/// while one knob said both.
#[test]
fn matmul_double_buffered_with_only_the_lhs_staged() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k, tile_edge) = (8usize, 8usize, 8usize, 4usize);
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, tile_edge), (N, tile_edge), (K, tile_edge)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::DOUBLE);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let dtype = f32::elem_type_native();
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .tile(&[tile_edge, tile_edge])
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::InPlace);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // The same answer the fully-staged double-buffered walk gives: residence moves bytes around,
    // it does not change what is computed.
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

/// Drives the staged lowering with a two-level partitioner stack `[l0, l1]`. `l1`'s
/// edge sizes the final tile (and the data tiling); the coarse `l0` drives launch geometry.
/// `stage` is the operands' stage-layout knob (the output is never staged).
#[allow(clippy::too_many_arguments)]
fn check_matmul_multilevel(
    m: usize,
    n: usize,
    k: usize,
    l0: Partitioner,
    l1: Partitioner,
    stage: StageStorage,
    residence: &[Residence],
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let final_edge = l1.edge(M);
    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(l0.clone())
        .with_partitioner(l1.clone());

    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    for &residence in residence {
        a_operand.stage(residence);
    }
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .tile(&[final_edge, final_edge])
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    for &residence in residence {
        b_operand.stage(residence);
    }
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .tile(&[final_edge, final_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[final_edge, final_edge])
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        TileArgLaunch::new(a.tensor_arg(1), a.spec().storage(stage)),
        TileArgLaunch::new(b.tensor_arg(1), b.spec().storage(stage)),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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

/// Drives the staged lowering `launch_staged_matmul` for `C = A @ B`. Every caller stages its one
/// level, so the inputs take [`Residence::Smem`] there.
fn check_matmul(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tile_edge = partitioner.edge(M);
    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());

    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .tile(&[tile_edge, tile_edge])
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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
/// from its partitioner's `Buffering` (here `.buffered(Buffering::SINGLE)` or `.buffered(Buffering::DOUBLE)`).
#[cube(launch)]
fn launch_staged_matmul<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mma(&a, &b);
}

/// The tensor-core kernel: promote the accumulator to its register form, zero it (the
/// classic `init_accumulator`), run the whole contraction on it, copy it back (the
/// epilogue).
#[cube(launch)]
fn launch_resident_matmul<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.accumulate(&a, LeafOp::Sum);
    acc.zero();
    acc.mma(&a, &b);
    c.copy_from(&acc);
}

/// Quantized `A` through the resident K walk: `A` is served via its quant arg, so `acc.mma`
/// dequantizes each K-stage's smem fill on its own — the fill recovers the storage element from
/// the scheme, so the kernel threads no `I` into the walk and the body is [`launch_resident_matmul`]
/// verbatim but for `A`'s served type. Tensor-core only.
#[cube(launch)]
fn launch_resident_matmul_quant<I: Numeric, E: Numeric, V: Size>(
    a: &QuantTileArg<'_, I, V>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.accumulate(&a, LeafOp::Sum);
    acc.zero();
    acc.mma(&a, &b);
    c.copy_from(&acc);
}

/// The CPU kernel: `c.zero()` then `c.mma(a, b)` (the production cpu_gemm body — the
/// register leaf accumulates in place, so the routine zeroes first); the default
/// `InPlace` residence selects the no-staging move. Operands are size-free —
/// vectorization is a launch concern, not threaded through the DSL.
#[cube(launch)]
fn launch_cpu_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// The promoted twin of [`launch_cpu_matmul`]: the register leaf's accumulator lifted out of
/// memory into its own block, contracted, and cast back down on drain.
#[cube(launch)]
fn launch_promoted_matmul<E: Numeric, EA: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.accumulate::<EA, _>(&a, LeafOp::Sum);
    acc.zero();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}

/// The register leaf contracts through a promoted block rather than through the output, so a
/// deep `K` keeps its partials in the accumulate element instead of round-tripping them
/// through the sink's on every visit.
#[test]
fn register_matmul_promoted_accumulator() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // One block per instance (a 1x1 partition at the leaf), K walked in four steps — every
    // step returns to the same promoted accumulator, which is the round trip this removes.
    let (m, n, k, edge) = (4usize, 4usize, 16usize, 4usize);
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, edge), (N, edge), (K, edge)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_promoted_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        dtype,
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

/// The promoted register accumulator under the two-level cube/plane space a real gemm composes,
/// with **vectorized** operands (rhs and output in 2-wide lines). This is the case that once
/// failed to compile on the CPU backend, when the block was allocated scalar and re-viewed as
/// lines; the block is now allocated at its vector element (`Array<Vector<T, RA>>`), so the
/// store is a real vector write and the numbers are right on every runtime.
#[test]
fn register_matmul_promoted_cube_plane() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (4usize, 4usize, 16usize);
    let (leaf_m, leaf_n, leaf_k) = (2usize, 2usize, 4usize);
    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, m))
                .axis(N, Cut::cube(CubeAxis::Y, n))
                .axis(K, seq(k))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::plane(leaf_m))
                .axis(N, Cut::plane(leaf_n))
                .axis(K, seq(leaf_k))
        })
        .build();

    let dtype = f32::elem_type_native();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    launch_promoted_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        // Rhs and output vectorized along N, as a real launch does: the tensor args stay
        // scalar-unit and the kernel's `Vector<E, V>` element carries the width.
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        dtype,
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
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

// ---- lined lhs: the (line, lane) K walk --------------------------------------

/// The register block walks `K` as (line, lane) so the lhs's within-line component is a comptime
/// index and `extract` names a fixed component. At a scalar lhs there is exactly one lane per line
/// and the split is invisible, so only a *lined* lhs exercises the fan-out. The two kernels below
/// are the scalar-lhs tests' twins with the lhs lined, one per caller of the shared walk.
#[cube(launch)]
fn launch_cpu_matmul_lined<E: Numeric, LV: Size, V: Size>(
    a: &TileArg<'_, E, LV>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// [`launch_cpu_matmul_lined`] through the promoted block instead of through the output.
#[cube(launch)]
fn launch_promoted_matmul_lined<E: Numeric, EA: Numeric, LV: Size, V: Size>(
    a: &TileArg<'_, E, LV>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.accumulate::<EA, _>(&a, LeafOp::Sum);
    acc.zero();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}

/// `A·B` off row-major `arange` operands: `lhs(i, p) = i·k + p`, `rhs(p, j) = p·n + j`.
fn arange_matmul_reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect()
}

/// A single-level space whose leaf takes the whole problem, the shape both lined-lhs tests drive.
fn lined_lhs_space(m: usize, n: usize, k: usize) -> Space {
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, m), (N, n), (K, k)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);
    Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner)
}

/// The memory-backed leaf with the lhs lined 2-wide along `K`: two lanes per K-line, each
/// reaching its element by a comptime `extract` rather than a dynamic one.
#[test]
fn register_matmul_lined_lhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_cpu_matmul_lined::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        2,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(arange_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// [`register_matmul_lined_lhs`] through the promoted block: same walk, same lanes, but the
/// accumulator never round-trips to the output between `K` steps.
#[test]
fn register_matmul_promoted_lined_lhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_promoted_matmul_lined::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        // Lhs 2-wide along K, rhs and output 2-wide along N: both the lane fan-out and the
        // block's own line width are off their scalar case at once.
        2,
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        dtype,
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(arange_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- folded step: both operands lined along K --------------------------------

/// Both operands lined along `K` with a scalar output: a step consumes a whole line, the block's
/// lanes are `K`-partials of one cell, and one horizontal fold collapses them. The rhs is declared
/// `[N, K]`, which is what puts its line on the contracted axis.
#[cube(launch)]
fn launch_matmul_folded<E: Numeric, AV: Size, BV: Size, CV: Size>(
    a: &TileArg<'_, E, AV>,
    b: &TileArg<'_, E, BV>,
    c: &TileArg<'_, E, CV>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// `A·Bᵀ` off row-major `arange` operands: `lhs(i, p) = i·k + p`, `rhs(j, p) = j·k + p`.
fn folded_matmul_reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (j * k + p)) as f32).sum()
        })
        .collect()
}

/// The 2-D nest at a folded step: four contracted values per `fma` instead of one.
#[test]
fn register_matmul_folded_step() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·Bᵀ` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_matmul_folded::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(64)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// [`register_matmul_folded_step`] with the block too big for the register budget: the same
/// numbers off the rolled body, whose local arrays are indexed at runtime.
#[test]
fn register_matmul_folded_step_rolled() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_matmul_folded::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(8)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The N-D nest at a folded step: two contracted axes, both operands lined along the faster of
/// them. The reduce nest steps by the served width, so each step lands on a line start.
#[test]
fn register_matmul_folded_step_two_contracted_axes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k1, k2) = (4usize, 4usize, 2usize, 4usize);
    let k = k1 * k2;
    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k1), (K2, k2)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, seq(m))
                .axis(N, seq(n))
                .axis(K, seq(k1))
                .axis(K2, seq(k2))
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K, K2]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K, K2]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    launch_matmul_folded::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(64)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The quantized folded step: the weight lined along `K` in packed `u32` words, the activation
/// lined along `K` in plain lines, and the decode sitting between the read and the `fma`.
#[cube(launch)]
fn launch_matmul_folded_quant<I: Numeric, E: Numeric, BV: Size>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, BV>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// A folded step whose lhs is packed `q8s` (4 values per word): the pack factor narrows the
/// *physical* line to one `u32` while the served line stays `pack` wide, so the same walk, gate
/// and fold run with a decode added on the read.
#[test]
fn register_matmul_folded_step_quant_q8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_folded_step_quant(client, QuantValue::Q8S, (4, 4, 8), 4);
}

/// The `q4s` twin: eight values per word, the brief's headline packing. Needs a device whose
/// vectors reach the factor, so it skips on WGSL-bound targets.
#[test]
fn register_matmul_folded_step_quant_q4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_folded_step_quant(client, QuantValue::Q4S, (4, 4, 16), 4);
}

/// Drive [`launch_matmul_folded_quant`] and check `C[i,j] = Σ_p q[i,p]·scale[i/bm]·B[j,p]`.
fn run_folded_step_quant(
    client: ComputeClient<TestRuntime>,
    value: QuantValue,
    (m, n, k): (usize, usize, usize),
    bm: usize,
) {
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let space = lined_lhs_space(m, n, k);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    launch_matmul_folded_quant::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        pack,
        QuantTileArgLaunch::new(
            a.tile.tensor_arg(1),
            a.scales_binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Read,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(64)),
        u32::elem_type_native(),
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a.q[i * k + p] as f32) * a.scale_values[i / bm] * ((j * k + p) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
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
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, 8), (N, 8)]);

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    cmma_roundtrip::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        input.arg(),
        output.arg(),
        space,
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
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = input.tile(space);
    let space = comptime!(a.space.clone());

    let mut a_smem = MemData::smem(
        comptime!(space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
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
        comptime!(StagePlan::in_place()),
    );
    c_smem.copy_from(&frag);
    sync_cube();

    let mut c = output.tile(space);
    c.copy_from(&c_smem);
}

/// A real 8×8×8 matmul through tensor cores: `C = A · B`, contracted by `cmma::execute`
/// on the cmma final space. Validates the fragment load → `execute` → store path against
/// the register reference. Tensor-core only — run with `cargo test-metal`.
#[test]
fn cmma_matmul_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
    if !require_cmma_8x8x8_f32(&client) {
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
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    // A: i8 quantized, with host values to build the reference.
    let a_dtype = ElemType::from_quant_value(scheme.value);
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

    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
    let e_dtype = f32::elem_type_native();

    cmma_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space,
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
    check_cmma_matmul_k_walk(16, Buffering::SINGLE);
}

/// The double-buffered variant: four K regions rotating through two smem slots, the
/// accumulator fragment resident across all of them.
#[test]
fn cmma_matmul_double_buffered_k_walk() {
    check_cmma_matmul_k_walk(32, Buffering::DOUBLE);
}

/// An odd region total (three K stages): the loop leaves the last region primed in slot 0;
/// the epilogue must publish and consume it.
#[test]
fn cmma_matmul_double_buffered_odd_k_walk() {
    check_cmma_matmul_k_walk(24, Buffering::DOUBLE);
}

/// The K walk staged into a plain strided stage (the legacy `sync_full_strided` storage):
/// the cmma window transport reads through the layout stack either way.
#[test]
fn cmma_matmul_staged_k_walk_strided_stage() {
    check_cmma_matmul_k_walk_v(16, Buffering::SINGLE, 1, StageStorage::Strided);
}

/// The leaf is the operands' statement and nothing else's: the partitioning says nothing about
/// it, so the memory instruction runs because all three operands declared it.
///
/// `mma_leaf` refuses a memory accumulator under a cmma leaf ("promote it first"), reading the
/// accumulator's own projected space, so this test fails loudly if an operand's declaration is
/// dropped rather than quietly running the other correct path.
#[test]
fn matmul_leaf_stated_by_operands() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (8usize, 8usize, 8usize);
    let seq = |edge| Cut::sequential(edge);
    let dtype = f32::elem_type_native();
    let mut ops = (
        Operand::new(&[M, K], dtype),
        Operand::new(&[K, N], dtype),
        Operand::new(&[M, N], dtype),
    );
    let space = Tiling::over(&mut ops, &[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, seq(4)).axis(N, seq(4)).axis(K, seq(4));
            o.0.stage(Residence::Smem);
            o.1.stage(Residence::Smem);
        })
        .build();

    let a = TileInput::builder(&client, space.project(ops.0.axes()))
        .operand(&ops.0)
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(ops.1.axes()))
        .operand(&ops.1)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(ops.2.axes()))
        .operand(&ops.2)
        .untiled()
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        TileArgLaunch::new(a.tensor_arg(1), a.spec()),
        TileArgLaunch::new(b.tensor_arg(1), b.spec()),
        TileArgLaunch::new(c.tensor_arg(1), c.spec()),
        space.with_instruction(Instruction::registers(16)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
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

fn check_cmma_matmul_k_walk(k: usize, buffering: Buffering) {
    check_cmma_matmul_k_walk_v(k, buffering, 1, StageStorage::Tiled)
}

/// The one level always stages, whatever it buffers: a cmma leaf cannot consume the global inputs
/// directly, so it first materializes them in shared memory.
fn check_cmma_matmul_k_walk_v(k: usize, buffering: Buffering, v: usize, stage: StageStorage) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, edge) = (8usize, 8usize, 8usize);
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, buffering, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    let dtype = f32::elem_type_native();
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .untiled()
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        v,
        TileArgLaunch::new(a.tensor_arg(1), a.spec().storage(stage)),
        TileArgLaunch::new(b.tensor_arg(1), b.spec().storage(stage)),
        c.arg(),
        space.with_instruction(instruction),
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

/// The manual/raw-mma instruction (`Instruction::Mma`): the raw-mma twin of `cmma_matmul_staged_k_walk` — the
/// same resident promote → zero → mma → drain kernel, but the contraction runs through
/// `MmaDefinition::execute` over register fragments rather than the cooperative `cmma::execute`.
/// Gated on the backend exposing the manual-mma feature (`features.matmul.mma`); uses the universal
/// manual transport (`MmaIOConfig::manual()`), so no `ldmatrix`/`stmatrix` path is taken. Run with
/// `cargo test-metal` / `test-cuda` on a backend that advertises manual mma.
#[test]
fn mma_matmul_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.mma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no manual mma (features.matmul.mma) support".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, edge) = (8usize, 8usize, 8usize, 8usize);
    let instruction = Instruction::Mma {
        io: MmaIOConfig::manual(),
    };
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    let dtype = f32::elem_type_native();
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .untiled()
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
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
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k, edge) = (16usize, 16usize, 32usize, 8usize);
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        // L0: the whole `16×16` output per cube, K walked in `8`-deep stages, double-buffered.
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l| {
            l.axis(M, Cut::sequential(m))
                .axis(N, Cut::sequential(n))
                .axis(K, Cut::sequential(edge))
        })
        // L1: the stage split one `8×8` fragment per plane.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::plane(edge))
                .axis(N, Cut::plane(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    let dtype = f32::elem_type_native();
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    a_operand.stage(Residence::InPlace);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .untiled()
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    b_operand.stage(Residence::InPlace);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
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
/// fragments, resident across a double-buffered K walk; the fragment level stays
/// `InPlace`, so the static walk reloads operand fragments per execute (no staging).
/// Tensor-core only; run with `cargo test-metal`.
#[test]
fn cmma_matmul_multi_fragment_partition() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let seq = |edge| Cut::sequential(edge);
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages.
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l| {
            l.axis(M, seq(m)).axis(N, seq(n)).axis(K, seq(stage_k))
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::plane(part))
                .axis(N, Cut::plane(part))
                .axis(K, seq(stage_k))
        })
        // L2: the partition level — 2×2 fragments per plane, 2 K sub-tiles.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, seq(i)).axis(N, seq(i)).axis(K, seq(i))
        })
        .build();

    let dtype = f32::elem_type_native();
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    a_operand.stage(Residence::Smem);
    a_operand.stage(Residence::InPlace);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .untiled()
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    b_operand.stage(Residence::InPlace);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
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
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);

    let mut a_smem_tile = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    a_smem_tile.copy_from(&a);

    let mut b_smem_tile = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    b_smem_tile.copy_from(&b);

    let mut c_smem_tile = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
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

/// Quantized `A`: gmem `I` (i8) dequantized into smem by the plain `copy_from`, which recovers
/// the storage element from the scheme on its own; `B`/`C` plain `E`. The cmma path then runs
/// entirely in `E`. Mirrors [`cmma_matmul`] otherwise.
#[cube(launch)]
fn cmma_matmul_quant<I: Numeric, E: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);

    let mut a_smem = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    a_smem.copy_from(&a);

    let mut b_smem = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    b_smem.copy_from(&b);

    let mut c_smem = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        comptime!(StagePlan::in_place()),
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

/// Block-quantized `A` (block along `M`): one flat `8×8` smem fill spans both scale blocks, the
/// per-line lookup picking each line's scale — `A`'s space needs no block sub-level. The cmma
/// fragment then reads the whole `8×8` smem. Validates block windowing into the matmul stage.
#[test]
fn cmma_matmul_quant_block_m_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
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
        .per_block([bm as u8, 8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
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

    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);

    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    cmma_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space,
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

/// Block-quantized `A` along `K` (the contraction axis): the scale changes partway through each
/// dot product, and the per-line lookup picks the right one mid-row. The case that matters for
/// quantized-weight matmul.
#[test]
fn cmma_matmul_quant_block_k_8x8x8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
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
        .per_block([8, bk as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
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

    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);

    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    cmma_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space,
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

/// Per-tensor-quantized `A` (i8) through the resident K walk, staged: `K = 16` runs in two
/// `8`-deep K regions, and each region's smem fill dequantizes `A` on its own — the same
/// `launch_resident_matmul_quant` body as the plain K walk. The self-describing fill in action.
/// Tensor-core only.
#[test]
fn cmma_matmul_quant_k_walk() {
    check_cmma_matmul_quant_k_walk(16, Buffering::SINGLE);
}

/// The same self-describing quant K walk driven double-buffered: both slots' fills dequantize.
#[test]
fn cmma_matmul_quant_double_buffered_k_walk() {
    check_cmma_matmul_quant_k_walk(32, Buffering::DOUBLE);
}

/// The manual-mma leaf decoding at the *read*: `DequantAt::Read` keeps `A`'s stage in its stored `i8`,
/// and the fragment load decodes each element through the quant-transparent matrix view. The cmma
/// twin of this test has no choice but `DequantAt::Load`, because its fragment load takes a raw window;
/// the manual transport addresses one element at a time, so it can decode. Same numbers, a stage
/// that is a quarter the size.
#[test]
fn mma_matmul_quant_until_read() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.mma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no manual mma (features.matmul.mma) support".to_string(),
        ))
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

    let (m, n, k, edge) = (8usize, 8usize, 16usize, 8usize);
    let instruction = Instruction::Mma {
        io: MmaIOConfig::manual(),
    };
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    let scale = 0.05f32;
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1, 1])
        .custom(vec![scale])
        .generate_without_host_data();

    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    launch_resident_matmul_quant::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]).residence(&[Residence::Smem]),
            scheme,
            DequantAt::Read,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a_host.get_f32(&[i, p]) * scale) * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The one level always stages, whatever it buffers: a cmma leaf cannot consume the global inputs
/// directly, so it first materializes them in shared memory.
fn check_cmma_matmul_quant_k_walk(k: usize, buffering: Buffering) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, edge) = (8usize, 8usize, 8usize); // K walked in `edge`-deep stages
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, buffering, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    let scale = 0.05f32;
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    // A: i8 quantized (m×k), with host values for the reference.
    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1, 1])
        .custom(vec![scale])
        .generate_without_host_data();

    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    launch_resident_matmul_quant::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]).residence(&[Residence::Smem]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale) · (p·n + j), p over all of K.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a_host.get_f32(&[i, p]) * scale) * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Block-M-quantized `A` through the resident K walk: one K stage stages the whole `M = 8`, which
/// spans two `bm = 4` scale blocks, so a single cooperative fill dequantizes across two scales —
/// the per-line scale lookup, not the one-scale-per-window assumption. `acc.mma` still just works.
/// Tensor-core only.
#[test]
fn cmma_matmul_quant_block_m_k_walk() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, edge, bm) = (8usize, 8usize, 16usize, 8usize, 4usize); // 2 M-blocks
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    // One scale per M-block, over the full K: block `[bm, k]`, scales shaped (m/bm, 1).
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..m / bm).map(|b| 0.05 * (b + 1) as f32).collect();

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![m / bm, 1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    launch_resident_matmul_quant::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]).residence(&[Residence::Smem]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale[i/bm]) · (p·n + j).
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a_host.get_f32(&[i, p]) * scale_vals[i / bm]) * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Block-K-quantized `A` through the resident K walk (the quantized-weight case): the scale
/// changes partway through each `8`-deep K stage (`bk = 4`), and it changes again between stages —
/// so the per-line scale lookup must fold in the stage's `window_start`. `acc.mma` just works.
/// Tensor-core only.
#[test]
fn cmma_matmul_quant_block_k_k_walk() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, edge, bk) = (8usize, 8usize, 16usize, 8usize, 4usize); // 4 K-blocks, 2 per stage
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    // One scale per K-block, over the full M: block `[m, bk]`, scales shaped (1, k/bk).
    let scheme = QuantScheme::default()
        .per_block([m as u8, bk as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..k / bk).map(|b| 0.05 * (b + 1) as f32).collect();

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1, k / bk])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    launch_resident_matmul_quant::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]).residence(&[Residence::Smem]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale[p/bk]) · (p·n + j).
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a_host.get_f32(&[i, p]) * scale_vals[p / bk]) * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Block-K-quantized `A` served in 2-wide lines: the blocks sit on the vectorized inner axis, so
/// a line's coordinate counts lines while its scale block is cut in elements — the widening
/// [`ScaleLayout`] does. Two lines per `bk = 4` block, so a stage's scale still changes mid-fill.
/// Tensor-core only.
#[test]
fn cmma_matmul_quant_block_k_k_walk_vectorized() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, edge, bk, v) = (8usize, 8usize, 16usize, 8usize, 4usize, 2usize);
    let instruction = Instruction::Cmma;
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(edge))
                .axis(N, Cut::sequential(edge))
                .axis(K, Cut::sequential(edge))
        })
        .build();

    // One scale per K-block, over the full M: block `[m, bk]`, scales shaped (1, k/bk).
    let scheme = QuantScheme::default()
        .per_block([m as u8, bk as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..k / bk).map(|b| 0.05 * (b + 1) as f32).collect();

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1, k / bk])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    b_operand.stage(Residence::Smem);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    launch_resident_matmul_quant::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        v,
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]).residence(&[Residence::Smem]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(instruction),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // C[i, j] = Σ_p (a_host[i, p] · scale[p/bk]) · (p·n + j).
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a_host.get_f32(&[i, p]) * scale_vals[p / bk]) * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Vectorized operands (2-wide lines) through the in-place path: gmem-only line-unit
/// addressing. Regression for the line-vs-scalar unit bug (worked on cubecl-cpu only).
#[test]
fn matmul_direct_vectorized() {
    check_matmul_vectorized(Buffering::SINGLE, &[], &[]);
}

/// The same walk with the operands staged instead: the cooperative fill moves lines through smem.
/// Regression for the line-vs-scalar unit bug. Its only difference from the direct case above is
/// the operands' residence, which is the whole point of stating it there.
#[test]
fn matmul_staged_vectorized() {
    check_matmul_vectorized(Buffering::SINGLE, &[Residence::Smem], &[Residence::Smem]);
}

/// The same operands through a depth-2 ring: each region's fill overlaps the previous region's
/// compute. Depth is the only difference from the staged case above.
#[test]
fn matmul_double_buffered_vectorized() {
    check_matmul_vectorized(Buffering::DOUBLE, &[Residence::Smem], &[Residence::Smem]);
}

/// Depth 3, which the single/double split could not express at all: two fills in flight over one
/// compute. Regression for a ring whose drain leaves more than one slot outstanding.
#[test]
fn matmul_triple_buffered_vectorized() {
    check_matmul_vectorized(Buffering::TRIPLE, &[Residence::Smem], &[Residence::Smem]);
}

/// A buffered level that *cuts* a promoted (fragment) accumulator: each region selects its own
/// block, so the ring's walk has to unroll and hand every region comptime coordinates.
///
/// The regression this guards is silent in both directions. `#[unroll(flag)]` only unrolls when
/// the macro sees `flag` as a comptime binding, and rolls the loop without complaint otherwise;
/// the lap arithmetic then has to fold, or the coordinates come out runtime even unrolled. Either
/// slip lands on `Tile::at`'s "must be walked with compile-time coordinates" panic. The other
/// unrolled shape, a `Residence::Register` stage, needs a fragment leaf and so only runs on tensor-core
/// hardware ([`cmma_matmul_staged_n_walk_partition`]); this one runs everywhere.
#[test]
fn matmul_buffered_walk_cutting_a_fragment_accumulator_unrolls() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        // L0: the whole output, K in two steps. `promote` mirrors this level's *sub-tile*, so the
        // accumulator's grid is only cut a level down.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
        })
        // L1: the 2x2 cut of that partition, buffered, with both operands staged.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(2))
                .axis(N, Cut::sequential(2))
                .axis(K, Cut::sequential(2))
        })
        .build();

    let dtype = f32::elem_type_native();
    let staged = [Residence::InPlace, Residence::Smem];
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    for &residence in &staged {
        a_operand.stage(residence);
    }
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .untiled()
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    for &residence in &staged {
        b_operand.stage(residence);
    }
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
        .uniform(4242, 10., 100.);

    launch_resident_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
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

/// A depth deeper than the walk has regions: the prologue runs out of regions to prime and every
/// consume drains. Regression for the ring's `region < total` guards.
#[test]
fn matmul_buffered_deeper_than_the_walk() {
    check_matmul_vectorized(Buffering::new(16), &[Residence::Smem], &[Residence::Smem]);
}

/// A depth-2 ring whose walk cuts only `M`: `rhs` spans `K`/`N` alone, so the walk never moves its
/// window. It is filled once above the loop and its buffer serves both slots
/// (`WindowMode::Reused`) --
/// the only sound way for two slots to reuse one buffer, and why a stage count is derived rather
/// than stated.
#[test]
fn matmul_double_buffered_with_a_fixed_operand() {
    check_matmul_dims_vectorized(
        (8, 4, 4),
        Buffering::DOUBLE,
        &[Residence::Smem],
        &[Residence::Smem],
    );
}

/// The same fixed operand three slots deep, so two slots reuse the first slot's buffer.
#[test]
fn matmul_triple_buffered_with_a_fixed_operand() {
    check_matmul_dims_vectorized(
        (8, 4, 4),
        Buffering::TRIPLE,
        &[Residence::Smem],
        &[Residence::Smem],
    );
}

/// One operand staged beside one read where it lies, at depth 2: the slot rendezvouses for the
/// staged one alone while the other is read where it lies, in every slot of the ring.
#[test]
fn matmul_double_buffered_mixed_residence_vectorized() {
    check_matmul_vectorized(Buffering::DOUBLE, &[Residence::Smem], &[Residence::InPlace]);
}

/// Every operand read where it lies, so the ring materializes nothing: the slots hold windows
/// alone, read at their own region. The depth is still the level's to state, which is the whole
/// point of running this level through the same walk as a staged one.
#[test]
fn matmul_all_in_place_double_buffered() {
    check_matmul_vectorized(
        Buffering::DOUBLE,
        &[Residence::InPlace],
        &[Residence::InPlace],
    );
}

/// The same level unbuffered: one slot, filled and consumed per region.
#[test]
fn matmul_all_in_place_single_buffered() {
    check_matmul_vectorized(
        Buffering::SINGLE,
        &[Residence::InPlace],
        &[Residence::InPlace],
    );
}

fn check_matmul_vectorized(
    buffering: Buffering,
    residence_a: &[Residence],
    residence_b: &[Residence],
) {
    check_matmul_dims_vectorized((8, 8, 8), buffering, residence_a, residence_b)
}

fn check_matmul_dims_vectorized(
    (m, n, k): (usize, usize, usize),
    buffering: Buffering,
    residence_a: &[Residence],
    residence_b: &[Residence],
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (edge, v) = (4usize, 2usize);
    let builder = Partitioner::row_major(
        ByAxis::new(&[(M, edge), (N, edge), (K, edge)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    );
    let partitioner = builder.buffered(buffering);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let dtype = f32::elem_type_native();
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    for &residence in residence_a {
        a_operand.stage(residence);
    }
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .operand(&a_operand)
        .untiled()
        .arange();
    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    for &residence in residence_b {
        b_operand.stage(residence);
    }
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        v,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
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
    check_cmma_matmul_k_walk_v(16, Buffering::SINGLE, 2, StageStorage::Tiled);
}

// ---- Quantized A through the register (plain-ALU) leaf --------------------------------
//
// Every other quant matmul above runs `acc.mma()` on tensor cores and skips where cmma is
// absent — which is everywhere the memory-bound GEMV actually lives. These pin the other
// leaf: the staged walk stages `A`'s *packed storage words* into smem (`Tile::copy_from`), and
// the software instruction dequantizes each read out of smem through `matrix_transparent` — no
// f32 inflation of the stage, no promotion, no cmma, no i8 needed for the packed cases (the
// binding is a `u32`).

/// The kernel: identical to [`launch_staged_matmul`] except `A` arrives storage-typed on its
/// quant arg, so the same lowering runs quantized or not.
#[cube(launch)]
fn launch_staged_matmul_quant<I: Numeric, E: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mma(&a, &b);
}

/// One staged level cutting `tm×tn×tk` register-leaf tiles — the shape `check_matmul`
/// drives, minus the storage tiling (operands stay plain strided).
/// The one-level partitioner both the staged and the direct-serve register-leaf tests walk. They
/// differ only in what the operands ask for there ([`Residence::Smem`] vs
/// [`Residence::InPlace`]), which is the whole distinction now that the level states no staging.
fn register_partitioner(tm: usize, tn: usize, tk: usize) -> Partitioner {
    Partitioner::row_major(
        ByAxis::new(&[(M, tm), (N, tn), (K, tk)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE)
}

/// Native i8 `A`, one scale per `bm`-row block, through the register leaf.
#[test]
fn register_matmul_quant_native_block_m() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, bm) = (8usize, 8usize, 8usize, 4usize);
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let q: Vec<f32> = (0..m * k)
        .map(|idx| a_host.get_f32(&[idx / k, idx % k]))
        .collect();

    let scale_vals: Vec<f32> = (0..m / bm).map(|g| 0.05 * (g + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), shape![m / bm, 1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    run_register_matmul_quant(
        client,
        (m, n, k),
        register_partitioner(4, 4, 4),
        &[Residence::Smem],
        a_input.binding().into_tensor_arg(),
        a_dtype,
        scheme,
        scales.binding().into_tensor_arg(),
        scale_vals,
        bm,
        q,
    );
}

/// Native i8 `A` served DIRECTLY through the register leaf (Keystone K): an all-`InPlace` plan
/// stages nothing, so the leaf reads i8 straight from gmem and scales per read. The native +
/// lhs-arm twin of the packed-rhs [`register_matmul_quant_rhs_direct_serve_gemv`]; together they
/// exercise every branch of the leaf's quant dispatch (lhs/rhs × native/packed).
#[test]
fn register_matmul_quant_native_direct_serve() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, bm) = (8usize, 8usize, 8usize, 4usize);
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let q: Vec<f32> = (0..m * k)
        .map(|idx| a_host.get_f32(&[idx / k, idx % k]))
        .collect();

    let scale_vals: Vec<f32> = (0..m / bm).map(|g| 0.05 * (g + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), shape![m / bm, 1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    run_register_matmul_quant(
        client,
        (m, n, k),
        register_partitioner(4, 4, 4),
        &[Residence::InPlace],
        a_input.binding().into_tensor_arg(),
        a_dtype,
        scheme,
        scales.binding().into_tensor_arg(),
        scale_vals,
        bm,
        q,
    );
}

/// Packed-u32 Q8S `A` (4 values per word along `K`), served in whole-word lines.
#[test]
fn register_matmul_quant_packed_q8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_register_matmul_quant_packed(client, (8, 8, 8), 4, QuantValue::Q8S, 4);
}

/// Packed-u32 Q4S `A` (8 values per word): the widest served line, so it needs a device
/// whose vectors reach the packing factor (cpu/cuda; WGSL-bound targets cap at 4).
#[test]
fn register_matmul_quant_packed_q4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_register_matmul_quant_packed(client, (8, 8, 16), 8, QuantValue::Q4S, 4);
}

/// Build a packed `A` spanning the scheme's signed range and run the register matmul.
fn run_register_matmul_quant_packed(
    client: ComputeClient<TestRuntime>,
    (m, n, k): (usize, usize, usize),
    tk: usize,
    value: QuantValue,
    bm: usize,
) {
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let a = TileInput::builder(&client, Space::new(&[(M, m), (K, k)]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();

    let a_dtype = u32::elem_type_native();
    let q: Vec<f32> = a.q.iter().map(|&v| v as f32).collect();
    run_register_matmul_quant(
        client,
        (m, n, k),
        register_partitioner(4, 4, tk),
        &[Residence::Smem],
        a.tile.tensor_arg(1),
        a_dtype,
        scheme,
        a.scales_binding().into_tensor_arg(),
        a.scale_values.clone(),
        bm,
        q,
    );
}

/// Drive [`launch_staged_matmul_quant`] and check `C[i,j] = Σ_p q[i,p]·scale[i/bm]·B[p,j]`.
/// `residence` is what the operands ask of the one level: [`Residence::Smem`] stages `A`'s packed
/// storage words into smem and dequantizes per read out of it, [`Residence::InPlace`] serves it
/// straight from gmem. Either way through the leaf's `matrix_transparent`, with no dequantized f32
/// stage.
#[allow(clippy::too_many_arguments)]
fn run_register_matmul_quant(
    client: ComputeClient<TestRuntime>,
    (m, n, k): (usize, usize, usize),
    plan: Partitioner,
    residence: &[Residence],
    a_arg: TensorArg<TestRuntime>,
    a_dtype: ElemType,
    scheme: QuantScheme,
    scales_arg: TensorArg<TestRuntime>,
    scale_vals: Vec<f32>,
    bm: usize,
    q: Vec<f32>,
) {
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(plan);

    let mut b_operand = Operand::new(&[K, N], f32::elem_type_native());
    for &residence in residence {
        b_operand.stage(residence);
    }
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();

    launch_staged_matmul_quant::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        QuantTileArgLaunch::new(
            a_arg,
            scales_arg,
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]).residence(residence),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        space.with_instruction(Instruction::registers(16)),
        a_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| q[i * k + p] * scale_vals[i / bm] * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- Quantized B (RHS) through the register leaf ---------------------------------------
//
// The gemv production shape: the *weight* is the streamed RHS — `(K, N) = (d_in, d_out)`,
// packed along `d_out` (the innermost axis) with one scale per `(k, N-group)` block
// (`[1, bn]`). A stays float. The RHS's served width drives the accumulator's line width
// in the register instruction, so `C` is launched at the same width.

/// [`launch_staged_matmul_quant`]'s mirror: `B` arrives storage-typed.
#[cube(launch)]
fn launch_staged_matmul_quant_rhs<I: Numeric, E: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &QuantTileArg<'_, I, Const<1>>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mma(&a, &b);
}

/// Packed-u32 Q8S `B` (4 values per word along `N`), scales `[1, bn]` — the exact scheme
/// family `metabolic`'s gemv ships (`q8s`, packed-u32, block scales along `d_out`).
#[test]
fn register_matmul_quant_rhs_packed_q8() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_register_matmul_quant_rhs(
        client,
        (8, 8, 8),
        register_partitioner(4, 4, 4),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        &[Residence::Smem],
        None,
    );
}

/// The `q4s` twin (8 values per word): needs 8-wide bindings, so cpu/cuda only.
#[test]
fn register_matmul_quant_rhs_packed_q4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_register_matmul_quant_rhs(
        client,
        (8, 16, 8),
        register_partitioner(4, 8, 4),
        QuantValue::Q4S,
        8,
        DequantAt::Read,
        &[Residence::Smem],
        None,
    );
}

/// The decode shape itself: a single activation row (`m = 1`) against the packed weight —
/// what every projection degenerates to during token-by-token generation.
#[test]
fn register_matmul_quant_rhs_gemv_row() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    run_register_matmul_quant_rhs(
        client,
        (1, 8, 8),
        register_partitioner(1, 4, 4),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        &[Residence::Smem],
        None,
    );
}

/// The decode shape spread across the device: `N` across cubes on `X`, the geometry a gemv
/// selector emits (`M = 1` leaves nothing else to spread).
#[test]
fn register_matmul_quant_rhs_gemv_row_multi_cube() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plan = Tiling::new()
        .extents(&[(M, 1), (N, 16), (K, 8)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::cube(CubeAxis::X, 4))
                .axis(K, Cut::sequential(4))
        })
        .build()
        .partitioner()
        .clone();
    run_register_matmul_quant_rhs(
        client,
        (1, 16, 8),
        plan,
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        &[Residence::Smem],
        None,
    );
}

/// Direct-serve the quantized RHS weight (Keystone K): an `InPlace` residence stages nothing,
/// so the register leaf reads the packed weight straight from gmem and dequantizes *per read*
/// through [`matrix_transparent`] — the sync-free `m = 1` decode path. The `_rhs_*` tests above are
/// all staged ([`Residence::Smem`]): they stage the weight's *packed words* into smem (plus its
/// scales) and dequantize per read out of smem. Same answer; direct avoids even the smem
/// round-trip.
#[test]
fn register_matmul_quant_rhs_direct_serve_gemv() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plan = Tiling::new()
        .extents(&[(M, 1), (N, 8), (K, 8)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
        })
        .build()
        .partitioner()
        .clone();
    run_register_matmul_quant_rhs(
        client,
        (1, 8, 8),
        plan,
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        &[Residence::InPlace],
        None,
    );
}

/// The Goal path: a staged ([`Residence::Smem`]) packed weight whose smem stage holds the *packed
/// u32 words*, not a dequantized f32 stage. A four-region K-walk (`k = 16`, `tk = 4`) with block
/// `[1, bn]` scales — distinct along K — so each region refills both the staged packed words and
/// the staged scales, and the leaf dequantizes per read out of smem via [`matrix_transparent`].
/// This is the batched weight-streaming case the change targets: the contrast to the f32-inflated
/// stage the cmma leaf still uses, and to the sync-free direct serve above.
#[test]
fn register_matmul_quant_rhs_staged_packed_smem() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plan = Tiling::new()
        .extents(&[(M, 4), (N, 8), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
        })
        .build()
        .partitioner()
        .clone();
    run_register_matmul_quant_rhs(
        client,
        (4, 8, 16),
        plan,
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        &[Residence::Smem],
        None,
    );
}

/// The same staged packed weight, decoded by the load instead of the read (`DequantAt::Load`): the
/// stage holds served values, so it costs the served-to-stored ratio in shared memory and the
/// decode happens once per element rather than per read. The fork a register leaf may take and a
/// cmma leaf is forced into; same numbers either way, which is the point of checking it.
#[test]
fn register_matmul_quant_rhs_staged_dequantized_smem() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plan = Tiling::new()
        .extents(&[(M, 4), (N, 8), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
        })
        .build()
        .partitioner()
        .clone();
    run_register_matmul_quant_rhs(
        client,
        (4, 8, 16),
        plan,
        QuantValue::Q8S,
        4,
        DequantAt::Load,
        &[Residence::Smem],
        None,
    );
}

/// Two-level through the staged `DequantAt::Read` path: the stage keeps the packed weight, and
/// `stage_scales` writes `global * local` into the smem scale grid, so the reads below see
/// effective one-level scales. The expectation carries the global scale, so a fold that never
/// happens (or happens twice) fails by that factor.
#[test]
fn register_matmul_quant_rhs_two_level_staged_packed_smem() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plan = Tiling::new()
        .extents(&[(M, 4), (N, 8), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
        })
        .build()
        .partitioner()
        .clone();
    run_register_matmul_quant_rhs(
        client,
        (4, 8, 16),
        plan,
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        &[Residence::Smem],
        Some(0.5),
    );
}

/// Two-level through the staged `DequantAt::Load` path: the fill dequantizes into the stage, so
/// the global scale folds in the gmem read itself and the stage carries plain served values.
#[test]
fn register_matmul_quant_rhs_two_level_staged_dequantized_smem() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plan = Tiling::new()
        .extents(&[(M, 4), (N, 8), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
        })
        .build()
        .partitioner()
        .clone();
    run_register_matmul_quant_rhs(
        client,
        (4, 8, 16),
        plan,
        QuantValue::Q8S,
        4,
        DequantAt::Load,
        &[Residence::Smem],
        Some(0.5),
    );
}

/// A cmma fragment loads at one element type, so it cannot decode as it reads: an operand staged
/// into registers, under a space whose instruction is cmma, while asking to stay quantized that
/// far, is refused before any kernel is built. The operand says where it lives and the space says
/// what consumes it there. Host-side, so it runs on every backend.
#[test]
#[should_panic(expected = "cannot decode as it reads")]
fn quant_until_read_refused_by_a_cmma_register_stage() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let scheme = QuantScheme::default()
        .per_block([1, 4], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
    let plan = Tiling::new()
        .extents(&[(M, 8), (N, 8), (K, 8)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(8))
                .axis(N, Cut::sequential(8))
                .axis(K, Cut::sequential(8))
        })
        .build()
        .partitioner()
        .clone();
    let space = Space::new(&[(M, 8), (N, 8), (K, 8)])
        .with_partitioner(plan)
        .with_instruction(Instruction::Cmma);
    let mut b_operand = Operand::new(&[K, N], u32::elem_type_native());
    b_operand.stage(Residence::Register);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .operand(&b_operand)
        .untiled()
        .packed(&scheme, DequantAt::Load)
        .arange();

    let launcher = space.launcher(&client);
    launcher
        .arg(b.tile.handle().binding())
        .subspace(&[K, N])
        .vectorize(scheme.num_quants())
        .operand(&b_operand)
        .quantized(&[b.scales_binding()], scheme, DequantAt::Read)
        .build();
}

/// Drive [`launch_staged_matmul_quant_rhs`] and check
/// `C[i,j] = Σ_p A[i,p] · q_b[p,j] · scale[p, j/bn]`.
#[allow(clippy::too_many_arguments)]
fn run_register_matmul_quant_rhs(
    client: ComputeClient<TestRuntime>,
    (m, n, k): (usize, usize, usize),
    plan: Partitioner,
    value: QuantValue,
    bn: usize,
    dequant_at: DequantAt,
    residence: &[Residence],
    global: Option<f32>,
) {
    // The data is minted against the one-level scheme either way: a two-level tensor holds the
    // same value and block-scale bytes, plus the global scale in its own binding.
    let mint_scheme = QuantScheme::default()
        .per_block([1, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let scheme = match global {
        Some(_) => mint_scheme.per_tensor(ScaleDtype::F32),
        None => mint_scheme,
    };
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(plan);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    // The weight and its per-(k, N-group) scales, minted together.
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .packed(&mint_scheme, dequant_at)
        .arange();
    let global_scale = global.map(|g| {
        TestInput::builder(client.clone(), shape![1])
            .custom(vec![g])
            .generate_without_host_data()
    });
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let b_dtype = u32::elem_type_native();
    let e_dtype = f32::elem_type_native();

    // Routine-like: the launcher derives geometry and argument wiring from the plan; the
    // quantized RHS goes through the source builder, which binds it at the storage width.
    let launcher = space.launcher(&client);
    let mut a_operand = Operand::new(&[M, K], f32::elem_type_native());
    for &r in residence {
        a_operand.stage(r);
    }
    let a_op = launcher.bind(&a_operand, a.handle().binding()).build();
    let mut b_operand = Operand::new(&[K, N], b_dtype);
    for &r in residence {
        b_operand.stage(r);
    }
    let b_src = launcher
        .bind(&b_operand, b.tile.handle().binding())
        .vectorize(pack);
    let mut scales = vec![b.scales_binding()];
    scales.extend(global_scale.map(|g| g.binding()));
    let b_op = b_src.quantized(&scales, scheme, dequant_at).build();
    // The register instruction lines the accumulator at the RHS's served width.
    let c_op = launcher
        .arg(c.handle().binding())
        .subspace(&[M, N])
        .vectorize(pack)
        .build();
    let b_arg = b_op.arg();
    launch_staged_matmul_quant_rhs::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        c_op.vector_size,
        a_op.arg(),
        b_arg,
        c_op.arg(),
        launcher
            .space()
            .clone()
            .with_instruction(Instruction::registers(16)),
        b_dtype,
        e_dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // A is arange over (m, k): a[i, p] = i·k + p.
    let sn = n / bn;
    let g = global.unwrap_or(1.0);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| {
                    ((i * k + p) as f32)
                        * (b.q[p * n + j] as f32)
                        * b.scale_values[p * sn + j / bn]
                        * g
                })
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}
