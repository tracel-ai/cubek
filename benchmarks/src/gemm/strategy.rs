use cubek::matmul::{
    launch::Strategy,
    routines::{
        BlueprintStrategy, TileSizeSelection, double_buffering::DoubleBufferingArgs,
        double_unit::DoubleUnitSelectionArgs, ordered_double_buffering::OrderedSelectionArgs,
        simple::SimpleArgs, simple_unit::SimpleUnitSelectionArgs,
    },
};

use crate::registry::ItemDescriptor;

#[derive(Clone, Copy, PartialEq, Eq)]
enum StrategyKind {
    SimpleCyclicCmma,
    SimpleCyclicCmmaMultiRows,
    DoubleTilewiseCmma,
    DoubleTilewiseCmmaSpecialized,
    OrderedDoubleCmma,
    SimpleUnitMin,
    SimpleUnitMax,
    DoubleUnitMin,
    DoubleUnitMax,
    SpecializedTmaMma,
    SpecializedCyclicMma,
    SpecializedStridedMma,
}

#[derive(Clone, Copy)]
struct StrategySpec {
    id: &'static str,
    label: &'static str,
    kind: StrategyKind,
}

const STRATEGIES: &[StrategySpec] = &[
    StrategySpec {
        id: "simple_cyclic_cmma",
        label: "SimpleCyclicCmma",
        kind: StrategyKind::SimpleCyclicCmma,
    },
    StrategySpec {
        id: "simple_cyclic_cmma_multirows",
        label: "SimpleCyclicCmma (multi rows)",
        kind: StrategyKind::SimpleCyclicCmmaMultiRows,
    },
    StrategySpec {
        id: "double_tilewise_cmma",
        label: "DoubleTilewiseCmma",
        kind: StrategyKind::DoubleTilewiseCmma,
    },
    StrategySpec {
        id: "double_tilewise_cmma_specialized",
        label: "DoubleTilewiseCmma (specialized)",
        kind: StrategyKind::DoubleTilewiseCmmaSpecialized,
    },
    StrategySpec {
        id: "ordered_double_cmma",
        label: "OrderedDoubleCmma (rc=8 rpp=2 pk=2)",
        kind: StrategyKind::OrderedDoubleCmma,
    },
    StrategySpec {
        id: "simple_unit_min",
        label: "Simple Unit (min tile)",
        kind: StrategyKind::SimpleUnitMin,
    },
    StrategySpec {
        id: "simple_unit_max",
        label: "Simple Unit (max tile)",
        kind: StrategyKind::SimpleUnitMax,
    },
    StrategySpec {
        id: "double_unit_min",
        label: "Double Unit (min tile)",
        kind: StrategyKind::DoubleUnitMin,
    },
    StrategySpec {
        id: "double_unit_max",
        label: "Double Unit (max tile)",
        kind: StrategyKind::DoubleUnitMax,
    },
    StrategySpec {
        id: "specialized_tma_mma",
        label: "Specialized TMA (mma)",
        kind: StrategyKind::SpecializedTmaMma,
    },
    StrategySpec {
        id: "specialized_cyclic_mma",
        label: "Specialized Cyclic (mma)",
        kind: StrategyKind::SpecializedCyclicMma,
    },
    StrategySpec {
        id: "specialized_strided_mma",
        label: "Specialized Strided (mma)",
        kind: StrategyKind::SpecializedStridedMma,
    },
];

pub fn strategies() -> Vec<ItemDescriptor> {
    STRATEGIES
        .iter()
        .map(|s| ItemDescriptor {
            id: s.id.to_string(),
            label: s.label.to_string(),
        })
        .collect()
}

pub(crate) fn strategy_for(id: &str) -> Option<Strategy> {
    let kind = STRATEGIES.iter().find(|s| s.id == id)?.kind;
    Some(match kind {
        StrategyKind::SimpleCyclicCmma => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
        }
        StrategyKind::SimpleCyclicCmmaMultiRows => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: true,
                ..Default::default()
            }))
        }
        StrategyKind::DoubleTilewiseCmma => {
            Strategy::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: false,
                ..Default::default()
            }))
        }
        StrategyKind::DoubleTilewiseCmmaSpecialized => {
            Strategy::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                ..Default::default()
            }))
        }
        StrategyKind::OrderedDoubleCmma => {
            Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
                ..Default::default()
            }))
        }
        StrategyKind::SimpleUnitMin => {
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            }))
        }
        StrategyKind::SimpleUnitMax => {
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            }))
        }
        StrategyKind::DoubleUnitMin => {
            Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            }))
        }
        StrategyKind::DoubleUnitMax => {
            Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            }))
        }
        StrategyKind::SpecializedTmaMma => {
            Strategy::SpecializedTmaMma(BlueprintStrategy::Inferred(().into()))
        }
        StrategyKind::SpecializedCyclicMma => {
            Strategy::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into()))
        }
        StrategyKind::SpecializedStridedMma => {
            Strategy::SpecializedStridedMma(BlueprintStrategy::Inferred(().into()))
        }
    })
}
