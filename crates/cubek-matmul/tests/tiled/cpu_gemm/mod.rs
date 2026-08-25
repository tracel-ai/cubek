mod inner_layout;
mod layouts;

use crate::harness::test_matmul_strategy;
use cubecl::{Runtime, frontend::Scalar, ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    routine::BlueprintStrategy,
    strategy::Strategy,
    tiled::{
        Strategy as Tiled,
        cpu_gemm::{CpuGemmBlueprint, CpuGemmStrategy, InstructionShape, PlaneGrid},
    },
};
use cubek_std::MatrixLayout;
use cubek_test_utils::skip_unless_cpu;

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
    if skip_unless_cpu(&client) {
        return;
    }
    let elems = MatmulGlobalElems {
        lhs: half::f16::elem_type_native(),
        rhs: half::f16::elem_type_native(),
        out: f32::elem_type_native(),
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
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// The `Inferred` strategy lets the routine pick the tile size. The heuristic sizes tiles
/// for L1 (large, and not divisors of these axes), so this exercises the runtime-looped
/// leaf path: a block too big to fully unroll loops at runtime instead.
#[test]
fn many_tiles_inferred_size() {
    let (batch, m, n, k) = (1, 64, 64, 128);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())).into(),
    );
}

/// A col-major (transposed) lhs: the layout the gemm routine's Col-Row variant used to
/// claim on CPU, and got wrong whenever `m` was not a multiple of the vector size
/// (burn#5304, `a.transpose().matmul(b)`). `m` here divides nothing.
#[test]
fn transposed_lhs_m_not_vector_multiple() {
    let (batch, m, n, k) = (1, 7, 8, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())).into(),
    );
}

/// The same transposed lhs at a size that actually tiles, batched.
#[test]
fn transposed_lhs_batched() {
    let (batch, m, n, k, tile_size) = (2, 33, 64, 64, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// No axis is a multiple of the tile, so every axis has a partial trailing tile. This
/// exercises edge masking (zero-padded input reads, predicated output writes).
#[test]
fn indivisible_all_axes() {
    let (batch, m, n, k, tile_size) = (1, 10, 10, 10, 4);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// Indivisible rectangular + batched, with K exact but M and N overhanging, so the
/// per-axis check flags differ across operands.
#[test]
fn indivisible_rectangular_batched() {
    let (batch, m, n, k, tile_size) = (2, 30, 20, 32, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// The `Inferred` heuristic on awkward primes: it no longer snaps the tile to a divisor,
/// so the chosen block overhangs and relies on masking.
#[test]
fn indivisible_inferred() {
    let (batch, m, n, k) = (1, 37, 41, 53);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())).into(),
    );
}

/// Matrix-vector (`n = 1`) with the `Inferred` heuristic: `n` is narrower than the SIMD
/// width, so the tile-size selector must not assume `n >= vw`. Regression for a `select()`
/// panic when clamping `tile_n` into `[vw, n]` with `n < vw`.
#[test]
fn matvec_inferred() {
    let (batch, m, n, k) = (1, 64, 1, 64);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())).into(),
    );
}

/// Narrow `n` (smaller than the SIMD width but `> 1`) under the `Inferred` heuristic: the
/// other side of the `n < vw` boundary, with `n` not a clean vector multiple.
#[test]
fn narrow_n_inferred() {
    let (batch, m, n, k) = (1, 32, 3, 48);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default())).into(),
    );
}

/// `rhs` unbatched (`[1]`) so it broadcasts across all of `lhs`'s batch: `rhs` omits the
/// batch axis, every cube reads the same matrix. `rhs` row-major exercises broadcast + `N`
/// vectorization together.
#[test]
fn broadcast_rhs_unbatched() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) = (shape![4], shape![1], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// `lhs` unbatched (`[1]`) broadcasts across `rhs`'s batch: the mirror case, so `lhs`
/// omits the batch axis instead. `rhs` col-major keeps it on the scalar path.
#[test]
fn broadcast_lhs_unbatched() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) = (shape![1], shape![4], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
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
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// A 2-D batch fully present on both sides (`[2, 3] @ [2, 3]`): no broadcast, but two batch
/// axes share cube-Z, exercising the multi-axis product on `Z`.
#[test]
fn batched_two_axes() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) =
        (shape![2, 3], shape![2, 3], 16, 16, 32, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// Broadcast crossed with edge masking: `rhs` broadcasts and no matrix axis divides the
/// tile, so partial tiles and the omitted batch axis are exercised together.
#[test]
fn broadcast_indivisible() {
    let (lhs_batches, rhs_batches, m, n, k, tile_size) = (shape![3], shape![1], 10, 14, 10, 4);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
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
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile_size,
                n: tile_size,
                k: tile_size,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }))
        .into(),
    );
}

/// A caller asking for an input register type of its own (tf32 registers off an f32 tensor)
/// is asking for a conversion this routine does not emit. It is rejected at setup rather
/// than run at the global type behind the caller's back.
#[test]
fn cpu_gemm_rejects_input_register_type() {
    use cubecl::{
        ir::{ElemType, FloatKind},
        prelude::*,
    };
    use cubek_matmul::{
        definition::MatmulSetupError,
        tiled::cpu_gemm::{WithLayout, launch_ref},
    };
    use cubek_std::InputBinding;
    use cubek_test_utils::TestInput;

    let client = TestRuntime::client(&Default::default());
    let f32t = f32::elem_type_native();
    let tensor = |seed| {
        TestInput::builder(client.clone(), shape![32, 32])
            .dtype(f32t)
            .layout(MatrixLayout::RowMajor)
            .uniform(seed, -1., 1.)
            .generate_without_host_data()
    };
    let (lhs, rhs, out) = (tensor(1234), tensor(5678), tensor(4242));

    // Everything f32 but the lhs register: the one field the kernel has no cast for.
    let mut dtypes = MatmulElems::from_single_dtype(f32t);
    dtypes.lhs_register = ElemType::Float(FloatKind::TF32);

    match launch_ref::<TestRuntime>(
        &client,
        WithLayout::strided_input(InputBinding::Normal(lhs.binding(), f32t)).unwrap(),
        WithLayout::strided_input(InputBinding::Normal(rhs.binding(), f32t)).unwrap(),
        WithLayout::strided_output(out.binding()).unwrap(),
        &Default::default(),
        &dtypes,
    ) {
        Err(MatmulSetupError::InvalidConfig(msg)) => {
            let msg = msg.to_string();
            assert!(
                msg.contains("Lhs") && msg.contains("TF32"),
                "wrong rejection: {msg}"
            );
        }
        other => panic!("expected a type rejection, got {other:?}"),
    }
}

/// The accumulator contracts at the stated type, not at the output buffer's.
///
/// Every `K` step after the first adds `2^-11` to a running `1.0`. That is half an `f16` ulp at
/// that magnitude, so an `f16` accumulator rounds every one of them away and answers `1.0`,
/// while the stated `f32` accumulator reaches `~5.0` and rounds once on drain. Nothing about
/// the shape is unusual, so only the accumulate width can move the result: a routine that
/// quietly contracted in the output's element would report a quarter of the answer.
#[test]
fn accumulator_holds_steps_the_output_element_cannot() {
    use cubecl::prelude::*;
    use cubek_matmul::{
        definition::MatmulGlobalElems as Globals,
        tiled::cpu_gemm::{WithLayout, launch_ref},
    };
    use cubek_std::InputBinding;
    use cubek_test_utils::{HostData, HostDataType, HostDataVec, TestInput};

    let (m, n, k, tile) = (8, 8, 8192, 8);
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
    let f16t = half::f16::elem_type_native();
    let step = 2f32.powi(-11);
    // One `1.0` row, then a long tail of steps too small to survive an `f16` running sum.
    let rhs_data: Vec<f32> = (0..k)
        .flat_map(|row| std::iter::repeat_n(if row == 0 { 1.0 } else { step }, n))
        .collect();
    let build = |shape: Vec<usize>, data: Vec<f32>| {
        TestInput::builder(client.clone(), shape)
            .dtype(f16t)
            .custom(data)
            .generate_without_host_data()
    };
    let lhs = build(vec![m, k], vec![1.0; m * k]);
    let rhs = build(vec![k, n], rhs_data);
    let out = build(vec![m, n], vec![0.0; m * n]);

    launch_ref::<TestRuntime>(
        &client,
        WithLayout::strided_input(InputBinding::Normal(lhs.binding(), f16t)).unwrap(),
        WithLayout::strided_input(InputBinding::Normal(rhs.binding(), f16t)).unwrap(),
        WithLayout::strided_output(out.clone().binding()).unwrap(),
        &BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile,
                n: tile,
                k: tile,
            },
            planes: PlaneGrid { m: 1, n: 1 },
        }),
        &MatmulElems::from_globals(&Globals {
            lhs: f16t,
            rhs: f16t,
            out: f16t,
        }),
    )
    .unwrap();

    let expected = 1.0 + (k - 1) as f32 * step;
    let actual = match HostData::from_tensor_handle(&client, out, HostDataType::F32).data {
        HostDataVec::F32(v) => v,
        other => panic!("expected f32 host data, got {other:?}"),
    };
    // The drain rounds to `f16` once, so allow one ulp there and nothing like the gap to `1.0`.
    for (i, got) in actual.iter().enumerate() {
        assert!(
            (got - expected).abs() < 0.01,
            "element {i}: got {got}, expected {expected} (an f16 accumulator answers 1.0)"
        );
    }
}
