use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_4D_TRANSPOSE: &str = "4d_swap_1_2_2_3";

pub struct ContiguousProblem {
    pub shape: Vec<usize>,
    pub dims: Vec<(usize, usize)>,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: PROBLEM_4D_TRANSPOSE.to_string(),
        label: "4D (16x16x512x512) swap (1,2)+(2,3)".to_string(),
    }]
}

pub(crate) fn problem_for(id: &str) -> Option<ContiguousProblem> {
    Some(match id {
        PROBLEM_4D_TRANSPOSE => ContiguousProblem {
            shape: vec![16, 16, 512, 512],
            dims: vec![(1, 2), (2, 3)],
        },
        _ => return None,
    })
}
