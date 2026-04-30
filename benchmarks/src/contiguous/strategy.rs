use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_DEFAULT: &str = "default";

pub struct ContiguousStrategy;

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: STRATEGY_DEFAULT.to_string(),
        label: "Default (into_contiguous)".to_string(),
    }]
}

pub(crate) fn strategy_for(id: &str) -> Option<ContiguousStrategy> {
    match id {
        STRATEGY_DEFAULT => Some(ContiguousStrategy),
        _ => None,
    }
}
