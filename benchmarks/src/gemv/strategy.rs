use cubek::matmul::{
    launch::Strategy,
    routines::{
        BlueprintStrategy, TileSizeSelection, simple::SimpleArgs,
        simple_unit::SimpleUnitSelectionArgs, vecmat_plane_parallel::GemvPlaneParallelStrategy,
        vecmat_unit_perpendicular::GemvUnitPerpendicularStrategy,
    },
};

use crate::registry::CatalogEntry;

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "gemv_unit_perpendicular",
            "Gemv Unit Perpendicular",
            Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
                GemvUnitPerpendicularStrategy {
                    target_num_planes: None,
                },
            )),
        ),
        CatalogEntry::new(
            "gemv_plane_parallel",
            "Gemv Plane Parallel",
            Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(GemvPlaneParallelStrategy {
                target_num_planes: None,
            })),
        ),
        CatalogEntry::new(
            "simple_vecmat",
            "Simple VecMat",
            Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
        ),
        CatalogEntry::new(
            "double_vecmat",
            "Double VecMat",
            Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
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
            "simple_cyclic_cmma",
            "Simple Cyclic CMMA",
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            })),
        ),
    ]
}
