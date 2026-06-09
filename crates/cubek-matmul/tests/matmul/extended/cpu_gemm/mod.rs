mod layouts;

use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    launch::Strategy,
    routines::{
        BlueprintStrategy,
        cpu_gemm::{CpuGemmBlueprint, CpuGemmStrategy},
    },
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

/// The shape of a CpuGemm test case: a `batch × (m, k) @ (k, n)` problem run with
/// square `tile_size` sub-tiles.
struct Dims {
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_size: usize,
}

#[test]
fn very_small_square() {
    let Dims {
        batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        batch: 1,
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
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
        })),
    );
}

#[test]
fn small_square() {
    let Dims {
        batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        batch: 1,
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
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
        })),
    );
}

#[test]
fn rectangular() {
    let Dims {
        batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        batch: 1,
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
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
        })),
    );
}

#[test]
fn single_tile() {
    let Dims {
        batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        batch: 1,
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
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
        })),
    );
}

/// The `Inferred` strategy lets the routine pick the tile size; this shape is
/// divisible by whatever edge the heuristic currently chooses.
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
        batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        batch: 4,
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
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
        })),
    );
}

#[test]
fn batched_rectangular() {
    let Dims {
        batch,
        m,
        n,
        k,
        tile_size,
    } = Dims {
        batch: 3,
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
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
        })),
    );
}
