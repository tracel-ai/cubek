use cubek_interpolate::interpolate_options::{InterpolateMode, InterpolateOptions};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_NEAREST: &str = "nearest";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: STRATEGY_NEAREST.to_string(),
        label: "Nearest".to_string(),
    }]
}

pub(crate) fn strategy_for(id: &str) -> Option<InterpolateOptions> {
    match id {
        STRATEGY_NEAREST => Some(InterpolateOptions::new(InterpolateMode::Nearest)),
        _ => None,
    }
}
