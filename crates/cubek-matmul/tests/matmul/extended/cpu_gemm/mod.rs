mod layouts;

use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    routines::{
        BlueprintStrategy,
        cpu_gemm::{CpuGemmBlueprint, CpuGemmStrategy, Instruction, PlaneGrid},
    },
    strategy::Strategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

/// The shape of a CpuGemm test case: `lhs_batch × (m, k) @ rhs_batch × (k, n)` run with
/// square `tile_size` sub-tiles. The two batches differ only for a broadcast case (one
/// side `1`); equal otherwise.
struct Dims {
    lhs_batch: usize,
    rhs_batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_size: usize,
}

/// Mixed precision: `f16` inputs, `f32` accumulate/output. Each operand reaches the leaf in
/// its own dtype and the inputs widen into `f32`, so the cast path runs through a real kernel.
#[test]
fn mixed_precision_f16_inputs_f32_acc() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 1,
        rhs_batch: 1,
        m: 32,
        n: 32,
        k: 64,
        tile_size: 8,
    };
    let client = TestRuntime::client(&Default::default());
    let elems = MatmulGlobalElems {
        lhs: half::f16::as_type_native_unchecked().storage_type(),
        rhs: half::f16::as_type_native_unchecked().storage_type(),
        out: f32::as_type_native_unchecked().storage_type(),
    };
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        // rhs/out row-major so the vectorized N path carries the cast on `V`-wide lines.
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

#[test]
fn very_small_square() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 1,
        rhs_batch: 1,
        m: 8,
        n: 8,
        k: 8,
        tile_size: 4,
    };
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

#[test]
fn small_square() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 1,
        rhs_batch: 1,
        m: 32,
        n: 32,
        k: 64,
        tile_size: 8,
    };
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

#[test]
fn rectangular() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 1,
        rhs_batch: 1,
        m: 48,
        n: 32,
        k: 64,
        tile_size: 16,
    };
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

#[test]
fn single_tile() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 1,
        rhs_batch: 1,
        m: 8,
        n: 8,
        k: 8,
        tile_size: 8,
    };
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// The `Inferred` strategy lets the routine pick the tile size. The heuristic sizes tiles
/// for L1 (large, and not divisors of these axes), so this exercises the runtime-looped
/// leaf path: a block too big to fully unroll loops at runtime instead.
#[test]
fn many_tiles_inferred_size() {
    let (batch, m, n, k) = (1, 64, 64, 128);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())),
    );
}

#[test]
fn batched_small() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 4,
        rhs_batch: 4,
        m: 16,
        n: 16,
        k: 32,
        tile_size: 8,
    };
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

#[test]
fn batched_rectangular() {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        lhs_batch: 3,
        rhs_batch: 3,
        m: 32,
        n: 48,
        k: 64,
        tile_size: 16,
    };
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// No axis is a multiple of the tile, so every axis has a partial trailing tile —
/// exercises edge masking (zero-padded input reads, predicated output writes).
#[test]
fn indivisible_all_axes() {
    let (batch, m, n, k, tile_size) = (1, 10, 10, 10, 4);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// Indivisible rectangular + batched, with K exact but M and N overhanging, so the
/// per-axis check flags differ across operands.
#[test]
fn indivisible_rectangular_batched() {
    let (batch, m, n, k, tile_size) = (2, 30, 20, 32, 8);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// The `Inferred` heuristic on awkward primes: it no longer snaps the tile to a divisor,
/// so the chosen block overhangs and relies on masking.
#[test]
fn indivisible_inferred() {
    let (batch, m, n, k) = (1, 37, 41, 53);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())),
    );
}

/// Matrix-vector (`n = 1`) with the `Inferred` heuristic: `n` is narrower than the SIMD
/// width, so the tile-size selector must not assume `n >= vw`. Regression for a `select()`
/// panic when clamping `tile_n` into `[vw, n]` with `n < vw`.
#[test]
fn matvec_inferred() {
    let (batch, m, n, k) = (1, 64, 1, 64);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())),
    );
}

/// Narrow `n` (smaller than the SIMD width but `> 1`) under the `Inferred` heuristic — the
/// other side of the `n < vw` boundary, with `n` not a clean vector multiple.
#[test]
fn narrow_n_inferred() {
    let (batch, m, n, k) = (1, 32, 3, 48);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())),
    );
}

/// `rhs` unbatched (`[1]`) so it broadcasts across all of `lhs`'s batch — `rhs` omits the
/// batch axis, every cube reads the same matrix. `rhs` row-major exercises broadcast + `N`
/// vectorization together.
#[test]
fn broadcast_rhs_unbatched() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) = (shape![4], shape![1], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// `lhs` unbatched (`[1]`) broadcasts across `rhs`'s batch — the mirror case, so `lhs`
/// omits the batch axis instead. `rhs` col-major keeps it on the scalar path.
#[test]
fn broadcast_lhs_unbatched() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) = (shape![1], shape![4], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// The genuine two-axis broadcast: `lhs [B0, 1]` and `rhs [1, B1]` give `out [B0, B1]`.
/// Each operand carries one batch axis and omits the other, so neither has the full batch;
/// the merge rebuilds `{B0, B1}` and both axes ride (share) cube-Z.
#[test]
fn broadcast_two_axes() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) =
        (shape![4, 1], shape![1, 3], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// A 2-D batch fully present on both sides (`[2, 3] @ [2, 3]`): no broadcast, but two batch
/// axes share cube-Z, exercising the multi-axis product on `Z`.
#[test]
fn batched_two_axes() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) =
        (shape![2, 3], shape![2, 3], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}

/// Broadcast crossed with edge masking: `rhs` broadcasts and no matrix axis divides the
/// tile, so partial tiles and the omitted batch axis are exercised together.
#[test]
fn broadcast_indivisible() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) = (shape![3], shape![1], 10, 14, 10, 4);
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile_size, tile_size, tile_size),
            planes: PlaneGrid::new(2, 2),
        })),
    );
}
