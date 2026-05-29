use cubek_test_utils::CatalogEntry;

use crate::launch::Strategy;
use crate::routines::{
    BlueprintStrategy, TileSizeSelection, double_buffering::DoubleBufferingArgs,
    double_unit::DoubleUnitSelectionArgs, gemm::GemmStrategy, mosaic::MosaicStrategy,
    ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
    simple_unit::SimpleUnitSelectionArgs,
};

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
            "mosaic",
            "Mosaic (tile-DSL CPU)",
            Strategy::Mosaic(MosaicStrategy::default()),
        ),
    ]
}
