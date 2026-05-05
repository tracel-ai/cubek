use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_DEFAULT: &str = "default";

pub struct InterpolateStrategy;

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: STRATEGY_DEFAULT.to_string(),
        label: "Default".to_string(),
    }]
}

pub(crate) fn strategy_for(id: &str) -> Option<InterpolateStrategy> {
    match id {
        STRATEGY_DEFAULT => Some(InterpolateStrategy),
        _ => None,
    }
}
