//! The multi-level half of the `gemm` catalogue.

use cubek_test_utils::CatalogEntry;

use crate::{
    multi_level::{
        Strategy as MultiLevel,
        routines::{
            TileSizeSelection,
            batch::{
                double_buffering::DoubleBufferingArgs, double_unit::DoubleUnitSelectionArgs,
                ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
                simple_unit::SimpleUnitSelectionArgs,
            },
            gemm::GemmStrategy,
        },
    },
    routine::BlueprintStrategy,
    strategy::Strategy,
};

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "simple_cyclic_cmma",
            "SimpleCyclicCmma",
            MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
            .into(),
        ),
        CatalogEntry::new(
            "simple_cyclic_cmma_multirows",
            "SimpleCyclicCmma (multi rows)",
            MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: true,
                ..Default::default()
            }))
            .into(),
        ),
        CatalogEntry::new(
            "double_tilewise_cmma",
            "DoubleTilewiseCmma",
            MultiLevel::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: false,
                ..Default::default()
            }))
            .into(),
        ),
        CatalogEntry::new(
            "double_tilewise_cmma_specialized",
            "DoubleTilewiseCmma (specialized)",
            MultiLevel::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                ..Default::default()
            }))
            .into(),
        ),
        CatalogEntry::new(
            "ordered_double_cmma",
            "OrderedDoubleCmma (rc=8 rpp=2 pk=2)",
            MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
                ..Default::default()
            }))
            .into(),
        ),
        CatalogEntry::new(
            "simple_unit_min",
            "Simple Unit (min tile)",
            MultiLevel::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            }))
            .into(),
        ),
        CatalogEntry::new(
            "simple_unit_max",
            "Simple Unit (max tile)",
            MultiLevel::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            }))
            .into(),
        ),
        CatalogEntry::new(
            "double_unit_min",
            "Double Unit (min tile)",
            MultiLevel::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            }))
            .into(),
        ),
        CatalogEntry::new(
            "double_unit_max",
            "Double Unit (max tile)",
            MultiLevel::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            }))
            .into(),
        ),
        CatalogEntry::new(
            "specialized_tma_mma",
            "Specialized TMA (mma)",
            MultiLevel::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())).into(),
        ),
        CatalogEntry::new(
            "specialized_cyclic_mma",
            "Specialized Cyclic (mma)",
            MultiLevel::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())).into(),
        ),
        CatalogEntry::new(
            "specialized_strided_mma",
            "Specialized Strided (mma)",
            MultiLevel::SpecializedStridedMma(BlueprintStrategy::Inferred(().into())).into(),
        ),
        CatalogEntry::new(
            "gemm",
            "Gemm",
            MultiLevel::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
                target_num_planes: None,
            }))
            .into(),
        ),
    ]
}
