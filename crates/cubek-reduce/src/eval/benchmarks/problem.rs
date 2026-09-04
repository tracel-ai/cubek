use cubecl::{features::TypeUsage, ir::ElemType, prelude::*};
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

/// The element type a problem's input and output tensors carry.
///
/// Real consumers reduce in f16, and the two are not the same measurement: the
/// kernel moves half the bytes and folds them in an f32 accumulator either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceBenchPrecision {
    F32,
    F16,
}

impl ReduceBenchPrecision {
    pub fn dtype(self) -> ElemType {
        match self {
            ReduceBenchPrecision::F32 => f32::elem_type_native(),
            ReduceBenchPrecision::F16 => half::f16::elem_type_native(),
        }
    }

    /// The suffix its rows' ids carry. An f32 row carries none: the catalogue
    /// tests and `CUBEK_BENCH_PROBLEMS` name those ids verbatim.
    fn suffix(self) -> &'static str {
        match self {
            ReduceBenchPrecision::F32 => "",
            ReduceBenchPrecision::F16 => "_f16",
        }
    }

    fn label(self) -> &'static str {
        match self {
            ReduceBenchPrecision::F32 => "",
            ReduceBenchPrecision::F16 => " [f16]",
        }
    }
}

/// The precisions this device can actually reduce in, so a runtime without f16
/// yields no f16 rows rather than a catalogue of failures.
pub fn precisions() -> Vec<ReduceBenchPrecision> {
    let client = cubecl::test_device().client();
    let mut precisions = vec![ReduceBenchPrecision::F32];
    if half::f16::supported_uses(&client).contains(TypeUsage::Arithmetic) {
        precisions.push(ReduceBenchPrecision::F16);
    }
    precisions
}

pub struct ReduceProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub config: ReduceOperationConfig,
    pub kind: ReduceBenchKind,
    pub precision: ReduceBenchPrecision,
}

pub fn problems() -> Vec<CatalogEntry<ReduceProblem>> {
    precisions().into_iter().flat_map(problems_at).collect()
}

fn problems_at(precision: ReduceBenchPrecision) -> Vec<CatalogEntry<ReduceProblem>> {
    let shape = || vec![32, 512, 4095];
    let id = |name: String| format!("{name}{}", precision.suffix());
    let label = |text: String| format!("{text}{}", precision.label());
    let problem = |config, kind| ReduceProblem {
        shape: shape(),
        axis: 2,
        config,
        kind,
        precision,
    };

    let mut entries = vec![
        CatalogEntry::new(
            id("sum_axis2_32x512x4095".to_string()),
            label("Sum axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperationConfig::Sum, ReduceBenchKind::Single),
        ),
        CatalogEntry::new(
            id("arg_topk1_axis2_32x512x4095".to_string()),
            label("ArgTopK(1) axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperationConfig::ArgTopK(1), ReduceBenchKind::Single),
        ),
        CatalogEntry::new(
            id("arg_topk2_axis2_32x512x4095".to_string()),
            label("ArgTopK(2) axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperationConfig::ArgTopK(2), ReduceBenchKind::Single),
        ),
        CatalogEntry::new(
            id("arg_topk3_axis2_32x512x4095".to_string()),
            label("ArgTopK(3) axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperationConfig::ArgTopK(3), ReduceBenchKind::Single),
        ),
    ];

    // The comparison that decides whether fusing pays off: the same top-k run
    // once as the two launches callers do today, once fused. The plain
    // single-output TopK(k) is kept per k so a regression in the values-only
    // path (which the fused work refactored) shows up on its own rather than
    // hiding inside the two-launch total.
    for k in [1, 2, 3, 5] {
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_single_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values only, 1 launch, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperationConfig::TopK(k), ReduceBenchKind::Single),
        ));
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_two_launch_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values+indices, 2 launches, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperationConfig::TopK(k), ReduceBenchKind::TwoLaunch),
        ));
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_fused_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values+indices, 1 fused launch, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperationConfig::TopK(k), ReduceBenchKind::Fused),
        ));
    }

    // Large `k`, where the selection network's `O(reduce_len * k)` shape shows.
    // The catalogue stopped at `k = 5`, which is why the cost of large `k` went
    // unnoticed: on a 5090 these run 0.18 ms at `k = 8` and 370 ms at `k = 256`.
    // `k = 32` is the interesting point — it is the largest that still fits the
    // unroll budget, and it is 5.8x faster for it. Fused only: what is being
    // measured is the accumulator, not the launch pattern.
    for k in [16, 32, 64, 128, 256] {
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_fused_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values+indices, 1 fused launch, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperationConfig::TopK(k), ReduceBenchKind::Fused),
        ));
    }

    // The same comparison for min and max, which reach `reduce_with_indices`
    // through their own collapsed instructions rather than the top-k one.
    for (name, text, config) in [
        ("max", "Max", ReduceOperationConfig::Max),
        ("min", "Min", ReduceOperationConfig::Min),
    ] {
        entries.push(CatalogEntry::new(
            id(format!("{name}_single_axis2_32x512x4095")),
            label(format!(
                "{text} values only, 1 launch, axis=2 (32x512x4095)"
            )),
            problem(config, ReduceBenchKind::Single),
        ));
        entries.push(CatalogEntry::new(
            id(format!("{name}_two_launch_axis2_32x512x4095")),
            label(format!(
                "{text} values+indices, 2 launches, axis=2 (32x512x4095)"
            )),
            problem(config, ReduceBenchKind::TwoLaunch),
        ));
        entries.push(CatalogEntry::new(
            id(format!("{name}_fused_axis2_32x512x4095")),
            label(format!(
                "{text} values+indices, 1 fused launch, axis=2 (32x512x4095)"
            )),
            problem(config, ReduceBenchKind::Fused),
        ));
    }

    entries
}
