use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_3D_MEDIUM: &str = "3d_32x512x2048";

pub struct UnaryProblem {
    pub shape: Vec<usize>,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: PROBLEM_3D_MEDIUM.to_string(),
        label: "3D (32x512x2048)".to_string(),
    }]
}

pub(crate) fn problem_for(id: &str) -> Option<UnaryProblem> {
    Some(match id {
        PROBLEM_3D_MEDIUM => UnaryProblem {
            shape: vec![32, 512, 2048],
        },
        _ => return None,
    })
}
