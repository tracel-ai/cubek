use std::sync::LazyLock;

use cubecl::{ir::ElemType, prelude::*};
use cubek_test_utils::CatalogEntry;

use crate::ReduceOperation;

/// Which launch pattern a problem measures.
///
/// Callers wanting a reduction's values *and* their indices have to run the
/// reduce twice today, so comparing [`Self::TwoLaunch`] against [`Self::Fused`]
/// on the same problem is what says whether fusing the two is actually worth it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceBenchKind {
    /// One `Reduce::execute` launch writing a single output.
    Single,
    /// Two `Reduce::execute` launches, the values operation then its `Arg*`, to get both halves.
    TwoLaunch,
    /// One `Reduce::execute_with_indices` launch writing both halves.
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
    pub fn suffix(self) -> &'static str {
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

/// The precisions this device can reduce in, so a runtime without f16 yields
/// no f16 rows. Computed once: a device's f16 support can't change, and every
/// catalogue listing and generated `f16()` test would otherwise re-query it.
static PRECISIONS: LazyLock<Vec<ReduceBenchPrecision>> = LazyLock::new(|| {
    let client = cubecl::test_device().client();
    let mut precisions = vec![ReduceBenchPrecision::F32];
    if cubek_test_utils::supports_f16_arithmetic(&client) {
        precisions.push(ReduceBenchPrecision::F16);
    }
    precisions
});

pub fn precisions() -> Vec<ReduceBenchPrecision> {
    PRECISIONS.clone()
}

pub struct ReduceProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
    pub kind: ReduceBenchKind,
    pub precision: ReduceBenchPrecision,
}

/// Whether `CUBEK_BENCH_TIER` asks for more than the default sweep, as it does
/// for interpolate. Unset or `light` keeps the cheap catalogue.
fn extended_tier() -> bool {
    !matches!(
        std::env::var("CUBEK_BENCH_TIER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "" | "light"
    )
}

pub fn problems() -> Vec<CatalogEntry<ReduceProblem>> {
    precisions().into_iter().flat_map(problems_at).collect()
}

fn problems_at(precision: ReduceBenchPrecision) -> Vec<CatalogEntry<ReduceProblem>> {
    let shape = || vec![32, 512, 4095];
    let id = |name: String| format!("{name}{}", precision.suffix());
    let label = |text: String| format!("{text}{}", precision.label());
    let problem = |operation, kind| ReduceProblem {
        shape: shape(),
        axis: 2,
        operation,
        kind,
        precision,
    };

    let mut entries = vec![
        CatalogEntry::new(
            id("sum_axis2_32x512x4095".to_string()),
            label("Sum axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperation::Sum, ReduceBenchKind::Single),
        ),
        CatalogEntry::new(
            id("arg_topk1_axis2_32x512x4095".to_string()),
            label("ArgTopK(1) axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperation::ArgTopK { k: 1 }, ReduceBenchKind::Single),
        ),
        CatalogEntry::new(
            id("arg_topk2_axis2_32x512x4095".to_string()),
            label("ArgTopK(2) axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperation::ArgTopK { k: 2 }, ReduceBenchKind::Single),
        ),
        CatalogEntry::new(
            id("arg_topk3_axis2_32x512x4095".to_string()),
            label("ArgTopK(3) axis=2 (32x512x4095)".to_string()),
            problem(ReduceOperation::ArgTopK { k: 3 }, ReduceBenchKind::Single),
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
            problem(ReduceOperation::TopK { k }, ReduceBenchKind::Single),
        ));
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_two_launch_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values+indices, 2 launches, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperation::TopK { k }, ReduceBenchKind::TwoLaunch),
        ));
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_fused_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values+indices, 1 fused launch, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperation::TopK { k }, ReduceBenchKind::Fused),
        ));
    }

    // Large `k`, where the selection network's `O(reduce_len * k)` shape shows.
    // `k = 32` is the largest that still fits the unroll budget, and 5.8x faster
    // for it. Fused only: this measures the accumulator, not the launch pattern.
    let mut large_k = vec![16, 32, 64];

    // A `k = 128` launch costs about a second, and reduce top-k stops being the
    // right algorithm near `k = 10`, so the tail is only swept when asked for.
    if extended_tier() {
        large_k.extend([128, 256]);
    }

    for k in large_k {
        entries.push(CatalogEntry::new(
            id(format!("topk{k}_fused_axis2_32x512x4095")),
            label(format!(
                "TopK({k}) values+indices, 1 fused launch, axis=2 (32x512x4095)"
            )),
            problem(ReduceOperation::TopK { k }, ReduceBenchKind::Fused),
        ));
    }

    // The same comparison for min and max, which reach `Reduce::execute_with_indices`
    // through their own collapsed instructions rather than the top-k one.
    for (name, text, operation) in [
        ("max", "Max", ReduceOperation::Max),
        ("min", "Min", ReduceOperation::Min),
    ] {
        entries.push(CatalogEntry::new(
            id(format!("{name}_single_axis2_32x512x4095")),
            label(format!(
                "{text} values only, 1 launch, axis=2 (32x512x4095)"
            )),
            problem(operation, ReduceBenchKind::Single),
        ));
        entries.push(CatalogEntry::new(
            id(format!("{name}_two_launch_axis2_32x512x4095")),
            label(format!(
                "{text} values+indices, 2 launches, axis=2 (32x512x4095)"
            )),
            problem(operation, ReduceBenchKind::TwoLaunch),
        ));
        entries.push(CatalogEntry::new(
            id(format!("{name}_fused_axis2_32x512x4095")),
            label(format!(
                "{text} values+indices, 1 fused launch, axis=2 (32x512x4095)"
            )),
            problem(operation, ReduceBenchKind::Fused),
        ));
    }

    // The shapes the default sweep spreads over, beside the one above: many short rows, few long
    // rows, rows across the contiguous dim (strided), and a whole tensor summed into one value.
    let spread = |name: &str, text: &str, shape: Vec<usize>, axis, operation| {
        let dims = shape
            .iter()
            .map(|extent| extent.to_string())
            .collect::<Vec<_>>()
            .join("x");
        CatalogEntry::new(
            id(format!("{name}_axis{axis}_{dims}")),
            label(format!("{text} axis={axis} ({dims})")),
            ReduceProblem {
                shape,
                axis,
                operation,
                kind: ReduceBenchKind::Single,
                precision,
            },
        )
    };
    entries.extend([
        spread(
            "sum_many_short_rows",
            "Sum, many short rows,",
            vec![262_144, 64],
            1,
            ReduceOperation::Sum,
        ),
        spread(
            "sum_few_long_rows",
            "Sum, few long rows,",
            vec![4, 4_194_304],
            1,
            ReduceOperation::Sum,
        ),
        spread(
            "sum_strided",
            "Sum, strided rows,",
            vec![32, 4095, 512],
            1,
            ReduceOperation::Sum,
        ),
        spread(
            "arg_topk1_strided",
            "ArgTopK(1), strided rows,",
            vec![32, 4095, 512],
            1,
            ReduceOperation::ArgTopK { k: 1 },
        ),
        spread(
            "sum_whole_tensor",
            "Sum, a whole tensor,",
            vec![1, 16_777_216],
            1,
            ReduceOperation::Sum,
        ),
    ]);

    // A decode sampler's row: one row of logits over a vocabulary, which one cube walks alone
    // unless the row is split across cubes. Its argmax, and a top-20 to sample among.
    let vocabulary = 151_936;
    let sampled = |operation, kind| ReduceProblem {
        shape: vec![1, vocabulary],
        axis: 1,
        operation,
        kind,
        precision,
    };
    entries.push(CatalogEntry::new(
        id(format!("arg_topk1_sampler_1x{vocabulary}")),
        label(format!("ArgTopK(1) sampler row, axis=1 (1x{vocabulary})")),
        sampled(ReduceOperation::ArgTopK { k: 1 }, ReduceBenchKind::Single),
    ));
    entries.push(CatalogEntry::new(
        id(format!("topk20_sampler_1x{vocabulary}")),
        label(format!(
            "TopK(20) values+indices, 1 fused launch, sampler row, axis=1 (1x{vocabulary})"
        )),
        sampled(ReduceOperation::TopK { k: 20 }, ReduceBenchKind::Fused),
    ));

    entries
}
