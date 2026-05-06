use cubek::reduce::components::instructions::ReduceOperationConfig;

use crate::registry::CatalogEntry;

pub struct ReduceProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub config: ReduceOperationConfig,
}

pub fn problems() -> Vec<CatalogEntry<ReduceProblem>> {
    let shape = || vec![32, 512, 4095];
    vec![
        CatalogEntry::new(
            "sum_axis2_32x512x4095",
            "Sum axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::Sum,
            },
        ),
        CatalogEntry::new(
            "arg_topk1_axis2_32x512x4095",
            "ArgTopK(1) axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::ArgTopK(1),
            },
        ),
        CatalogEntry::new(
            "arg_topk2_axis2_32x512x4095",
            "ArgTopK(2) axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::ArgTopK(2),
            },
        ),
        CatalogEntry::new(
            "arg_topk3_axis2_32x512x4095",
            "ArgTopK(3) axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::ArgTopK(3),
            },
        ),
    ]
}
