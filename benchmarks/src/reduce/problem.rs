use cubek::reduce::components::instructions::ReduceOperationConfig;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_SUM_AXIS2: &str = "sum_axis2_32x512x4095";
pub const PROBLEM_ARG_TOPK1_AXIS2: &str = "arg_topk1_axis2_32x512x4095";
pub const PROBLEM_ARG_TOPK2_AXIS2: &str = "arg_topk2_axis2_32x512x4095";
pub const PROBLEM_ARG_TOPK3_AXIS2: &str = "arg_topk3_axis2_32x512x4095";

pub struct ReduceProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub config: ReduceOperationConfig,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_SUM_AXIS2.to_string(),
            label: "Sum axis=2 (32x512x4095)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_ARG_TOPK1_AXIS2.to_string(),
            label: "ArgTopK(1) axis=2 (32x512x4095)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_ARG_TOPK2_AXIS2.to_string(),
            label: "ArgTopK(2) axis=2 (32x512x4095)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_ARG_TOPK3_AXIS2.to_string(),
            label: "ArgTopK(3) axis=2 (32x512x4095)".to_string(),
        },
    ]
}

pub(crate) fn problem_for(id: &str) -> Option<ReduceProblem> {
    let shape = vec![32, 512, 4095];
    Some(match id {
        PROBLEM_SUM_AXIS2 => ReduceProblem {
            shape,
            axis: 2,
            config: ReduceOperationConfig::Sum,
        },
        PROBLEM_ARG_TOPK1_AXIS2 => ReduceProblem {
            shape,
            axis: 2,
            config: ReduceOperationConfig::ArgTopK(1),
        },
        PROBLEM_ARG_TOPK2_AXIS2 => ReduceProblem {
            shape,
            axis: 2,
            config: ReduceOperationConfig::ArgTopK(2),
        },
        PROBLEM_ARG_TOPK3_AXIS2 => ReduceProblem {
            shape,
            axis: 2,
            config: ReduceOperationConfig::ArgTopK(3),
        },
        _ => return None,
    })
}
