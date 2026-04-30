use cubek::attention::{
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_UNIT: &str = "unit_inferred";
pub const STRATEGY_BLACKBOX_ACCELERATED: &str = "blackbox_accelerated_inferred";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_UNIT.to_string(),
            label: "Unit (inferred)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_BLACKBOX_ACCELERATED.to_string(),
            label: "Blackbox accelerated (inferred, np=1 sq=1 skv=1)".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<Strategy> {
    match id {
        STRATEGY_UNIT => Some(Strategy::Unit(BlueprintStrategy::Inferred(()))),
        STRATEGY_BLACKBOX_ACCELERATED => Some(Strategy::BlackboxAccelerated(
            BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
                num_planes: 1,
                seq_q: 1,
                seq_kv: 1,
            }),
        )),
        _ => None,
    }
}
