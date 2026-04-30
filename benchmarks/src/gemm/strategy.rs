use cubek::matmul::{
    launch::Strategy,
    routines::{
        BlueprintStrategy, double_buffering::DoubleBufferingArgs,
        ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
    },
};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_SIMPLE_CYCLIC_CMMA: &str = "simple_cyclic_cmma";
pub const STRATEGY_SIMPLE_CYCLIC_CMMA_MULTIROWS: &str = "simple_cyclic_cmma_multirows";
pub const STRATEGY_DOUBLE_TILEWISE_CMMA: &str = "double_tilewise_cmma";
pub const STRATEGY_DOUBLE_TILEWISE_CMMA_SPECIALIZED: &str = "double_tilewise_cmma_specialized";
pub const STRATEGY_ORDERED_DOUBLE_CMMA: &str = "ordered_double_cmma";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_SIMPLE_CYCLIC_CMMA.to_string(),
            label: "SimpleCyclicCmma".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SIMPLE_CYCLIC_CMMA_MULTIROWS.to_string(),
            label: "SimpleCyclicCmma (multi rows)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_DOUBLE_TILEWISE_CMMA.to_string(),
            label: "DoubleTilewiseCmma".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_DOUBLE_TILEWISE_CMMA_SPECIALIZED.to_string(),
            label: "DoubleTilewiseCmma (specialized)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_ORDERED_DOUBLE_CMMA.to_string(),
            label: "OrderedDoubleCmma (rc=8 rpp=2 pk=2)".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<Strategy> {
    Some(match id {
        STRATEGY_SIMPLE_CYCLIC_CMMA => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
        }
        STRATEGY_SIMPLE_CYCLIC_CMMA_MULTIROWS => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: true,
                ..Default::default()
            }))
        }
        STRATEGY_DOUBLE_TILEWISE_CMMA => {
            Strategy::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: false,
                ..Default::default()
            }))
        }
        STRATEGY_DOUBLE_TILEWISE_CMMA_SPECIALIZED => {
            Strategy::DoubleTilewiseCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                ..Default::default()
            }))
        }
        STRATEGY_ORDERED_DOUBLE_CMMA => {
            Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
                ..Default::default()
            }))
        }
        _ => return None,
    })
}
