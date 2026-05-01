use cubek::matmul::{
    launch::Strategy,
    routines::{
        BlueprintStrategy, TileSizeSelection, simple::SimpleArgs,
        simple_unit::SimpleUnitSelectionArgs, vecmat_plane_parallel::GemvPlaneParallelStrategy,
        vecmat_unit_perpendicular::GemvUnitPerpendicularStrategy,
    },
};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_GEMV_UNIT_PERPENDICULAR: &str = "gemv_unit_perpendicular";
pub const STRATEGY_GEMV_PLANE_PARALLEL: &str = "gemv_plane_parallel";
pub const STRATEGY_SIMPLE_VECMAT: &str = "simple_vecmat";
pub const STRATEGY_DOUBLE_VECMAT: &str = "double_vecmat";
pub const STRATEGY_SIMPLE_UNIT_MIN: &str = "simple_unit_min";
pub const STRATEGY_SIMPLE_UNIT_MAX: &str = "simple_unit_max";
pub const STRATEGY_SIMPLE_CYCLIC_CMMA: &str = "simple_cyclic_cmma";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_GEMV_UNIT_PERPENDICULAR.to_string(),
            label: "Gemv Unit Perpendicular".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_GEMV_PLANE_PARALLEL.to_string(),
            label: "Gemv Plane Parallel".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SIMPLE_VECMAT.to_string(),
            label: "Simple VecMat".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_DOUBLE_VECMAT.to_string(),
            label: "Double VecMat".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SIMPLE_UNIT_MIN.to_string(),
            label: "Simple Unit (min tile)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SIMPLE_UNIT_MAX.to_string(),
            label: "Simple Unit (max tile)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SIMPLE_CYCLIC_CMMA.to_string(),
            label: "Simple Cyclic CMMA".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<Strategy> {
    Some(match id {
        STRATEGY_GEMV_UNIT_PERPENDICULAR => Strategy::GemvUnitPerpendicular(
            BlueprintStrategy::Inferred(GemvUnitPerpendicularStrategy {
                target_num_planes: None,
            }),
        ),
        STRATEGY_GEMV_PLANE_PARALLEL => {
            Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(GemvPlaneParallelStrategy {
                target_num_planes: None,
            }))
        }
        STRATEGY_SIMPLE_VECMAT => Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
        STRATEGY_DOUBLE_VECMAT => Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
        STRATEGY_SIMPLE_UNIT_MIN => {
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            }))
        }
        STRATEGY_SIMPLE_UNIT_MAX => {
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            }))
        }
        STRATEGY_SIMPLE_CYCLIC_CMMA => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
        }
        _ => return None,
    })
}
