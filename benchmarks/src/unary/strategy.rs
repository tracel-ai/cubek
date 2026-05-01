use cubecl::prelude::VectorSize;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_VEC1: &str = "vec1";
pub const STRATEGY_VEC4: &str = "vec4";
pub const STRATEGY_VEC8: &str = "vec8";

pub struct UnaryStrategy {
    pub vectorization: VectorSize,
}

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_VEC1.to_string(),
            label: "Vec1".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_VEC4.to_string(),
            label: "Vec4".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_VEC8.to_string(),
            label: "Vec8".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<UnaryStrategy> {
    Some(match id {
        STRATEGY_VEC1 => UnaryStrategy { vectorization: 1 },
        STRATEGY_VEC4 => UnaryStrategy { vectorization: 4 },
        STRATEGY_VEC8 => UnaryStrategy { vectorization: 8 },
        _ => return None,
    })
}
