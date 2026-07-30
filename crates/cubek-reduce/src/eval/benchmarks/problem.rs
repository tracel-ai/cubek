use cubek_test_utils::CatalogEntry;

use crate::components::instructions::ReduceOperationConfig;

/// Which launch pattern a problem measures.
///
/// Callers wanting a reduction's values *and* their indices have to run the
/// reduce twice today, so comparing [`Self::TwoLaunch`] against [`Self::Fused`]
/// on the same problem is what says whether fusing the two is actually worth it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceBenchKind {
    /// One `reduce` launch writing a single output.
    Single,
    /// Two `reduce` launches, the values config then its `Arg*`, to get both halves.
    TwoLaunch,
    /// One `reduce_with_indices` launch writing both halves.
    Fused,
}

pub struct ReduceProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub config: ReduceOperationConfig,
    pub kind: ReduceBenchKind,
}

pub fn problems() -> Vec<CatalogEntry<ReduceProblem>> {
    let shape = || vec![32, 512, 4095];

    let mut entries = vec![
        CatalogEntry::new(
            "sum_axis2_32x512x4095",
            "Sum axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::Sum,
                kind: ReduceBenchKind::Single,
            },
        ),
        CatalogEntry::new(
            "arg_topk1_axis2_32x512x4095",
            "ArgTopK(1) axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::ArgTopK(1),
                kind: ReduceBenchKind::Single,
            },
        ),
        CatalogEntry::new(
            "arg_topk2_axis2_32x512x4095",
            "ArgTopK(2) axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::ArgTopK(2),
                kind: ReduceBenchKind::Single,
            },
        ),
        CatalogEntry::new(
            "arg_topk3_axis2_32x512x4095",
            "ArgTopK(3) axis=2 (32x512x4095)",
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::ArgTopK(3),
                kind: ReduceBenchKind::Single,
            },
        ),
    ];

    // The comparison that decides whether fusing pays off: the same top-k run
    // once as the two launches callers do today, once fused. The plain
    // single-output TopK(k) is kept per k so a regression in the values-only
    // path (which the fused work refactored) shows up on its own rather than
    // hiding inside the two-launch total.
    for k in [1, 2, 3, 5] {
        entries.push(CatalogEntry::new(
            format!("topk{k}_single_axis2_32x512x4095"),
            format!("TopK({k}) values only, 1 launch, axis=2 (32x512x4095)"),
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::TopK(k),
                kind: ReduceBenchKind::Single,
            },
        ));
        entries.push(CatalogEntry::new(
            format!("topk{k}_two_launch_axis2_32x512x4095"),
            format!("TopK({k}) values+indices, 2 launches, axis=2 (32x512x4095)"),
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::TopK(k),
                kind: ReduceBenchKind::TwoLaunch,
            },
        ));
        entries.push(CatalogEntry::new(
            format!("topk{k}_fused_axis2_32x512x4095"),
            format!("TopK({k}) values+indices, 1 fused launch, axis=2 (32x512x4095)"),
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config: ReduceOperationConfig::TopK(k),
                kind: ReduceBenchKind::Fused,
            },
        ));
    }

    // The same comparison for min and max, which reach `reduce_with_indices`
    // through their own collapsed instructions rather than the top-k one.
    for (name, label, config) in [
        ("max", "Max", ReduceOperationConfig::Max),
        ("min", "Min", ReduceOperationConfig::Min),
    ] {
        entries.push(CatalogEntry::new(
            format!("{name}_single_axis2_32x512x4095"),
            format!("{label} values only, 1 launch, axis=2 (32x512x4095)"),
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config,
                kind: ReduceBenchKind::Single,
            },
        ));
        entries.push(CatalogEntry::new(
            format!("{name}_two_launch_axis2_32x512x4095"),
            format!("{label} values+indices, 2 launches, axis=2 (32x512x4095)"),
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config,
                kind: ReduceBenchKind::TwoLaunch,
            },
        ));
        entries.push(CatalogEntry::new(
            format!("{name}_fused_axis2_32x512x4095"),
            format!("{label} values+indices, 1 fused launch, axis=2 (32x512x4095)"),
            ReduceProblem {
                shape: shape(),
                axis: 2,
                config,
                kind: ReduceBenchKind::Fused,
            },
        ));
    }

    entries
}
