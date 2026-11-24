use std::sync::LazyLock;

use cubecl::client::Client;
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
    pub fn to_dtype(self) -> ElemType {
        match self {
            ReduceBenchPrecision::F32 => f32::elem_type_native(),
            ReduceBenchPrecision::F16 => half::f16::elem_type_native(),
        }
    }

    /// Whether `client`'s device reduces in it.
    pub fn is_offered_on(self, client: &Client) -> bool {
        match self {
            ReduceBenchPrecision::F32 => true,
            ReduceBenchPrecision::F16 => cubek_test_utils::supports_f16_arithmetic(client),
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
}

impl ReduceBenchPrecision {
    fn label(self) -> &'static str {
        match self {
            ReduceBenchPrecision::F32 => "",
            ReduceBenchPrecision::F16 => " [f16]",
        }
    }
}

/// The precisions this device can reduce in, so a runtime without f16 yields
/// no f16 rows. Computed once: a device's f16 support can't change, and every
/// catalogue listing would otherwise re-query it.
static PRECISIONS: LazyLock<Vec<ReduceBenchPrecision>> = LazyLock::new(|| {
    let client = cubecl::test_device().client();
    [ReduceBenchPrecision::F32, ReduceBenchPrecision::F16]
        .into_iter()
        .filter(|precision| precision.is_offered_on(&client))
        .collect()
});

fn precisions() -> &'static [ReduceBenchPrecision] {
    &PRECISIONS
}

/// One reduction of the catalogue: the shape it reads, what it computes, and how it is launched.
pub struct ReduceProblem {
    /// The catalogued input's extent along each axis, read by axis (Position Usage) and walked in
    /// their order (Arrival Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
    pub kind: ReduceBenchKind,
    pub precision: ReduceBenchPrecision,
}

impl ReduceProblem {
    /// How far a kernel's output may sit from the reference before it is wrong.
    ///
    /// A selection instruction returns a value taken out of the input, or its
    /// coordinate, so any slack at all leaves the comparison unable to fail. Only
    /// the accumulating ones round, and over tens of millions of f32 elements some
    /// kernels accumulate noticeable noise; tightening those belongs in the
    /// per-routine integration tests.
    pub fn tolerance(&self) -> f32 {
        match self.operation {
            ReduceOperation::Sum | ReduceOperation::Mean | ReduceOperation::Prod => 1.0,
            ReduceOperation::Max
            | ReduceOperation::Min
            | ReduceOperation::MaxAbs
            | ReduceOperation::ArgMax
            | ReduceOperation::ArgMin
            | ReduceOperation::TopK { .. }
            | ReduceOperation::ArgTopK { .. }
            | ReduceOperation::Any
            | ReduceOperation::All => 0.0,
        }
    }
}

/// Whether `CUBEK_BENCH_TIER` asks for more than the default sweep, as it does
/// for interpolate.
fn extended_tier() -> bool {
    is_extended_tier(&std::env::var("CUBEK_BENCH_TIER").unwrap_or_default())
}

/// Whether `tier` is more than the default sweep: unset or `light` keeps the cheap catalogue.
fn is_extended_tier(tier: &str) -> bool {
    !matches!(tier.to_lowercase().as_str(), "" | "light")
}

/// Every problem, at each precision this device reduces in.
pub(crate) fn problems() -> Vec<CatalogEntry<ReduceProblem>> {
    precisions().iter().copied().flat_map(problems_at).collect()
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
    // `k = 64` is the largest that still fits the unroll budget. Fused only: this measures the accumulator, not the launch pattern.
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
    let spread = |SpreadShape {
                      name,
                      text,
                      shape,
                      axis,
                      operation,
                  }: SpreadShape| {
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
        spread(SpreadShape {
            name: "sum_many_short_rows",
            text: "Sum, many short rows,",
            shape: vec![262_144, 64],
            axis: 1,
            operation: ReduceOperation::Sum,
        }),
        spread(SpreadShape {
            name: "sum_few_long_rows",
            text: "Sum, few long rows,",
            shape: vec![4, 4_194_304],
            axis: 1,
            operation: ReduceOperation::Sum,
        }),
        spread(SpreadShape {
            name: "sum_strided",
            text: "Sum, strided rows,",
            shape: vec![32, 4095, 512],
            axis: 1,
            operation: ReduceOperation::Sum,
        }),
        spread(SpreadShape {
            name: "arg_topk1_strided",
            text: "ArgTopK(1), strided rows,",
            shape: vec![32, 4095, 512],
            axis: 1,
            operation: ReduceOperation::ArgTopK { k: 1 },
        }),
        spread(SpreadShape {
            name: "sum_whole_tensor",
            text: "Sum, a whole tensor,",
            shape: vec![1, 16_777_216],
            axis: 1,
            operation: ReduceOperation::Sum,
        }),
    ]);

    // Every other operation, along the contiguous dim and across it: what each one's accumulator
    // costs beside the sum's and the arg top-1's above.
    for (name, text, operation) in [
        ("mean", "Mean", ReduceOperation::Mean),
        ("prod", "Prod", ReduceOperation::Prod),
        ("max", "Max", ReduceOperation::Max),
        ("min", "Min", ReduceOperation::Min),
        ("max_abs", "MaxAbs", ReduceOperation::MaxAbs),
        ("arg_max", "ArgMax", ReduceOperation::ArgMax),
        ("arg_min", "ArgMin", ReduceOperation::ArgMin),
        ("any", "Any", ReduceOperation::Any),
        ("all", "All", ReduceOperation::All),
    ] {
        entries.push(spread(SpreadShape {
            name: &format!("{name}_rows"),
            text: &format!("{text}, rows,"),
            shape: vec![32, 512, 4095],
            axis: 2,
            operation,
        }));
        entries.push(spread(SpreadShape {
            name: &format!("{name}_strided"),
            text: &format!("{text}, strided rows,"),
            shape: vec![32, 4095, 512],
            axis: 1,
            operation,
        }));
    }

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

/// One shape of the default sweep: its catalogue id `name`, its label `text`, and the reduction
/// along `axis` by `operation`.
struct SpreadShape<'a> {
    pub name: &'a str,
    pub text: &'a str,
    /// The swept input's extent along each axis, read by axis (Position Usage) and walked in their
    /// order (Arrival Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn no_tier_is_the_default_sweep() {
        assert!(!is_extended_tier(""));
    }

    #[test]
    fn the_light_tier_is_the_default_sweep() {
        assert!(!is_extended_tier("Light"));
    }

    #[test]
    fn another_tier_is_more_than_the_default_sweep() {
        assert!(is_extended_tier("heavy"));
    }

    #[test]
    fn f16_is_stored_in_f16() {
        let dtype = ReduceBenchPrecision::F16.to_dtype();
        assert_eq!(dtype, half::f16::elem_type_native());
    }

    #[test]
    fn f32_is_offered_everywhere() {
        let client = cubecl::test_device().client();
        assert!(ReduceBenchPrecision::F32.is_offered_on(&client));
    }

    #[test]
    fn an_f32_row_carries_no_suffix() {
        assert_eq!(ReduceBenchPrecision::F32.suffix(), "");
    }

    #[test]
    fn an_f16_rows_ids_carry_its_suffix() {
        let suffix = ReduceBenchPrecision::F16.suffix();
        let rows = problems_at(ReduceBenchPrecision::F16);
        assert!(rows.iter().all(|row| row.id.ends_with(suffix)));
    }

    #[test]
    fn an_f16_rows_labels_name_it() {
        let rows = problems_at(ReduceBenchPrecision::F16);
        assert!(rows.iter().all(|row| row.label.ends_with("[f16]")));
    }

    #[test]
    fn every_problems_id_is_its_own() {
        let problems = problems();
        let ids: HashSet<&str> = problems.iter().map(|problem| problem.id.as_ref()).collect();
        assert_eq!(ids.len(), problems.len());
    }

    /// A problem of `operation`.
    fn problem_of(operation: ReduceOperation) -> ReduceProblem {
        ReduceProblem {
            shape: vec![4, 8],
            axis: 1,
            operation,
            kind: ReduceBenchKind::Single,
            precision: ReduceBenchPrecision::F32,
        }
    }

    #[test]
    fn a_selection_is_exact() {
        assert_eq!(problem_of(ReduceOperation::ArgMax).tolerance(), 0.0);
    }

    #[test]
    fn a_sum_may_round() {
        assert!(problem_of(ReduceOperation::Sum).tolerance() > 0.0);
    }
}
