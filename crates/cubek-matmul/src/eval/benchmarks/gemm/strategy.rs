use cubek_test_utils::CatalogEntry;

use crate::routines::{
    BlueprintStrategy, TileSizeSelection,
    batch::{
        double_buffering::DoubleBufferingArgs, double_unit::DoubleUnitSelectionArgs,
        ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
        simple_unit::SimpleUnitSelectionArgs,
    },
    cpu_gemm::{CpuGemmBlueprint, Instruction, PlaneGrid},
    gemm::GemmStrategy,
};
use crate::strategy::Strategy;

fn cpu_gemm_forced(
    tag: &'static str,
    label: &'static str,
    tile: usize,
    plane_m: usize,
    plane_n: usize,
) -> CatalogEntry<Strategy> {
    CatalogEntry::new(
        tag,
        label,
        Strategy::CpuGemm(BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction {
                m: tile,
                n: tile,
                k: tile,
            },
            planes: PlaneGrid {
                m: plane_m,
                n: plane_n,
            },
        })),
    )
}

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "simple_cyclic_cmma",
            "SimpleCyclicCmma",
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            })),
        ),
        CatalogEntry::new(
            "simple_cyclic_cmma_multirows",
            "SimpleCyclicCmma (multi rows)",
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: true,
                ..Default::default()
            })),
        ),
        CatalogEntry::new(
            "double_tilewise_cmma",
            "DoubleTilewiseCmma",
            Strategy::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: false,
                ..Default::default()
            })),
        ),
        CatalogEntry::new(
            "double_tilewise_cmma_specialized",
            "DoubleTilewiseCmma (specialized)",
            Strategy::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                ..Default::default()
            })),
        ),
        CatalogEntry::new(
            "ordered_double_cmma",
            "OrderedDoubleCmma (rc=8 rpp=2 pk=2)",
            Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
                ..Default::default()
            })),
        ),
        CatalogEntry::new(
            "simple_unit_min",
            "Simple Unit (min tile)",
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            })),
        ),
        CatalogEntry::new(
            "simple_unit_max",
            "Simple Unit (max tile)",
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            })),
        ),
        CatalogEntry::new(
            "double_unit_min",
            "Double Unit (min tile)",
            Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            })),
        ),
        CatalogEntry::new(
            "double_unit_max",
            "Double Unit (max tile)",
            Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            })),
        ),
        CatalogEntry::new(
            "specialized_tma_mma",
            "Specialized TMA (mma)",
            Strategy::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())),
        ),
        CatalogEntry::new(
            "specialized_cyclic_mma",
            "Specialized Cyclic (mma)",
            Strategy::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())),
        ),
        CatalogEntry::new(
            "specialized_strided_mma",
            "Specialized Strided (mma)",
            Strategy::SpecializedStridedMma(BlueprintStrategy::Inferred(().into())),
        ),
        CatalogEntry::new(
            "gemm",
            "Gemm",
            Strategy::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
                target_num_planes: None,
            })),
        ),
        CatalogEntry::new(
            "cpu_gemm",
            "CpuGemm (tile-DSL CPU)",
            Strategy::CpuGemm(BlueprintStrategy::default()),
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
    ]
}
