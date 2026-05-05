use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_NEAREST_UPSAMPLE: &str = "nearest_upsample_4x4_to_10x10";

pub struct InterpolateProblem {
    pub input_shape: [usize; 4],
    pub output_size: [usize; 2],
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: PROBLEM_NEAREST_UPSAMPLE.to_string(),
        label: "Nearest upsample (b=2 h=4 w=4 c=2 -> 10x10)".to_string(),
    }]
}

pub(crate) fn problem_for(id: &str) -> Option<InterpolateProblem> {
    Some(match id {
        PROBLEM_NEAREST_UPSAMPLE => InterpolateProblem {
            input_shape: [2, 4, 4, 2],
            output_size: [10, 10],
        },
        _ => return None,
    })
}
