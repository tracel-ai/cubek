//! The tiled half of the `gemm` catalogue: the same problems as the multi-level
//! entries, so the two architectures are compared on one table.

use cubek_test_utils::CatalogEntry;

use crate::{
    routine::BlueprintStrategy,
    strategy::Strategy,
    tiled::{
        Strategy as Tiled,
        cmma::CmmaStrategy,
        cpu_gemm::{CpuGemmBlueprint, InstructionShape, PlaneGrid},
    },
};

fn cpu_gemm_forced(
    tag: &'static str,
    label: &'static str,
    tile: usize,
    plane_m: usize,
    plane_n: usize,
) -> CatalogEntry<Strategy> {
    cpu_gemm_leaf(tag, label, tile, tile, tile, plane_m, plane_n)
}

/// Forced CpuGemm with an explicit (non-square) leaf and plane grid. Used for the fast-core
/// scaling study: fix the register-fit leaf and vary the worker-thread count.
#[allow(clippy::too_many_arguments)]
fn cpu_gemm_leaf(
    tag: &'static str,
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
    plane_m: usize,
    plane_n: usize,
) -> CatalogEntry<Strategy> {
    CatalogEntry::new(
        tag,
        label,
        Tiled::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape { m, n, k },
            planes: PlaneGrid {
                m: plane_m,
                n: plane_n,
            },
        }))
        .into(),
    )
}

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "cpu_gemm",
            "CpuGemm (tile-DSL CPU)",
            Tiled::CpuGemm(BlueprintStrategy::default()).into(),
        ),
        CatalogEntry::new(
            "cmma",
            "Cmma (tile-DSL)",
            Tiled::Cmma(BlueprintStrategy::default()).into(),
        ),
        CatalogEntry::new(
            "cmma_tma",
            "Cmma (tile-DSL, TMA)",
            Tiled::Cmma(BlueprintStrategy::Inferred(CmmaStrategy::tma())).into(),
        ),
        cpu_gemm_forced(
            "cpu_gemm_t64",
            "CpuGemm (forced 64³, maskless on 512)",
            64,
            2,
            2,
        ),
        cpu_gemm_forced(
            "cpu_gemm_t48",
            "CpuGemm (forced 48³, masked on 512)",
            48,
            2,
            2,
        ),
        cpu_gemm_forced(
            "cpu_gemm_t32",
            "CpuGemm (forced 32³, maskless on 512)",
            32,
            2,
            2,
        ),
        // Plane-scaling study at a fixed 64³ leaf: 1 → 2 → 4 → 8 worker threads per cube.
        cpu_gemm_forced("cpu_gemm_p1", "CpuGemm (64³, 1 plane)", 64, 1, 1),
        cpu_gemm_forced("cpu_gemm_p2", "CpuGemm (64³, 2 planes)", 64, 2, 1),
        cpu_gemm_forced("cpu_gemm_p4", "CpuGemm (64³, 4 planes)", 64, 2, 2),
        cpu_gemm_forced("cpu_gemm_p8", "CpuGemm (64³, 8 planes)", 64, 4, 2),
        // Fast-core scaling: fix the register-fit leaf (2×32×64, no spill, the optimized
        // instruction) and scale the worker threads 1 → 16. Measures how the *fast* core spreads.
        cpu_gemm_leaf(
            "cpu_gemm_fast_p1",
            "CpuGemm (fast leaf, 1 thread)",
            2,
            32,
            64,
            1,
            1,
        ),
        cpu_gemm_leaf(
            "cpu_gemm_fast_p4",
            "CpuGemm (fast leaf, 4 threads)",
            2,
            32,
            64,
            2,
            2,
        ),
        cpu_gemm_leaf(
            "cpu_gemm_fast_p8",
            "CpuGemm (fast leaf, 8 threads)",
            2,
            32,
            64,
            4,
            2,
        ),
        cpu_gemm_leaf(
            "cpu_gemm_fast_p16",
            "CpuGemm (fast leaf, 16 threads)",
            2,
            32,
            64,
            8,
            2,
        ),
    ]
}
