use cubek_test_utils::CatalogEntry;

use crate::{
    multi_level::{
        Strategy as MultiLevel,
        routines::{
            TileSizeSelection,
            batch::{simple::SimpleArgs, simple_unit::SimpleUnitSelectionArgs},
            gemm::GemmStrategy,
            gemv_unit_perpendicular::GemvUnitPerpendicularStrategy,
        },
    },
    routine::BlueprintStrategy,
    strategy::Strategy,
};

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "gemv_unit_perpendicular",
            "Gemv Unit Perpendicular",
            MultiLevel::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
                GemvUnitPerpendicularStrategy {
                    target_num_planes: None,
                },
            ))
            .into(),
        ),
        CatalogEntry::new(
            "gemm",
            "Gemm",
            MultiLevel::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
                target_num_planes: None,
            }))
            .into(),
        ),
        CatalogEntry::new(
            "simple_vecmat",
            "Simple VecMat",
            MultiLevel::SimpleVecMat(BlueprintStrategy::Inferred(().into())).into(),
        ),
        CatalogEntry::new(
            "double_vecmat",
            "Double VecMat",
            MultiLevel::DoubleVecMat(BlueprintStrategy::Inferred(().into())).into(),
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
            "simple_cyclic_cmma",
            "Simple Cyclic CMMA",
            MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
            .into(),
        ),
    ]
}
