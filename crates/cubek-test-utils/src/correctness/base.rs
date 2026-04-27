use std::ops::Range;

use crate::{
    correctness::color_printer::ColorPrinter,
    test_mode::{TestMode, current_test_mode},
    {HostData, ValidationResult},
};

/// Maximum number of individual mismatches to include in a `Fail` reason
/// before we stop recording details and only update the aggregate stats.
const DEFAULT_MAX_REPORTED_MISMATCHES: usize = 8;

/// Check if two tensors are approximately equal
pub fn assert_equals_approx(
    // One of the tensor to compare
    lhs: &HostData,
    // One of the tensor to compare
    rhs: &HostData,
    // Maximum absolute difference between two values
    epsilon: f32,
) -> ValidationResult {
    assert_equals_approx_inner(lhs, rhs, epsilon, None)
}

/// Check if two tensors are approximately equal
/// Within the given slice, if some
pub fn assert_equals_approx_in_slice(
    // One of the tensor to compare
    lhs: &HostData,
    // One of the tensor to compare
    rhs: &HostData,
    // Maximum absolute difference between two values
    epsilon: f32,
    // If some, will only check values within the slice shape
    slice: Vec<Range<usize>>,
) -> ValidationResult {
    assert_equals_approx_inner(lhs, rhs, epsilon, Some(slice))
}

/// Check if two tensors are approximately equal
/// Within the given slice, if some
fn assert_equals_approx_inner(
    // One of the tensor to compare
    lhs: &HostData,
    // One of the tensor to compare
    rhs: &HostData,
    // Maximum absolute difference between two values
    epsilon: f32,
    // If some, will only check values within the slice shape
    slice: Option<Vec<Range<usize>>>,
) -> ValidationResult {
    if lhs.shape != rhs.shape {
        return ValidationResult::Fail(format!(
            "Shape mismatch: got {:?}, expected {:?}",
            lhs.shape, rhs.shape,
        ));
    }

    let shape = &lhs.shape;
    let test_mode = current_test_mode();

    let print_visitor = matches!(test_mode, TestMode::Print { .. });
    let mut summary_visitor = SummaryCollector::new(DEFAULT_MAX_REPORTED_MISMATCHES);
    let mut printer_visitor: Option<SafePrinter> = match &test_mode {
        TestMode::Print { filter, .. } => Some(SafePrinter {
            inner: ColorPrinter::new(filter.clone()),
            shape_len: shape.len(),
        }),
        _ => None,
    };

    let mut visitor = TeeVisitor {
        summary: &mut summary_visitor,
        printer: printer_visitor.as_mut().map(|v| v as &mut dyn CompareVisitor),
    };

    let test_failed = compare_tensors(
        lhs,
        rhs,
        shape,
        epsilon,
        &mut visitor,
        &mut Vec::new(),
        slice.as_deref(), // pass slice as Option<&[usize]>
    );

    // Enforce filter rank only if the test failed and we would print
    if test_failed {
        if let TestMode::Print { filter, .. } = &test_mode
            && !filter.is_empty()
            && filter.len() != shape.len()
        {
            return ValidationResult::Error(format!(
                "Print mode activated with invalid filter rank. Got {:?}, expected {:?}",
                filter.len(),
                shape.len()
            ));
        }

        return ValidationResult::Fail(summary_visitor.report(shape, print_visitor));
    }

    ValidationResult::Pass
}

#[derive(Debug)]
pub(crate) enum ElemStatus {
    Correct { got: f32, delta: f32, epsilon: f32 },
    Wrong(WrongStatus),
}

#[derive(Debug)]
pub(crate) enum WrongStatus {
    GotWrongValue {
        got: f32,
        expected: f32,
        delta: f32,
        epsilon: f32,
    },
    ExpectedNan {
        got: f32,
    },
    GotNan {
        expected: f32,
    },
}

pub(crate) trait CompareVisitor {
    fn visit(&mut self, index: &[usize], status: ElemStatus);
}

struct SafePrinter {
    inner: ColorPrinter,
    shape_len: usize,
}

impl CompareVisitor for SafePrinter {
    fn visit(&mut self, index: &[usize], status: ElemStatus) {
        // Only forward to the inner printer if filter rank is valid
        if self.inner.filter.is_empty() || self.inner.filter.len() == self.shape_len {
            self.inner.visit(index, status);
        } else {
            // skip printing silently
        }
    }
}

/// Collects up to `max_reported` individual mismatches plus aggregate stats
/// (total compared, total mismatched, max/sum of |Δ|, worst index) to build a
/// detailed failure message without truncating useful debug info.
pub(crate) struct SummaryCollector {
    pub max_reported: usize,
    pub mismatches: Vec<(Vec<usize>, WrongStatus)>,
    pub total: usize,
    pub mismatched: usize,
    pub max_abs_delta: f32,
    pub sum_abs_delta: f64,
    pub worst_index: Option<Vec<usize>>,
}

impl SummaryCollector {
    fn new(max_reported: usize) -> Self {
        Self {
            max_reported,
            mismatches: Vec::new(),
            total: 0,
            mismatched: 0,
            max_abs_delta: 0.0,
            sum_abs_delta: 0.0,
            worst_index: None,
        }
    }

    fn record_delta(&mut self, index: &[usize], delta: f32) {
        if delta.is_finite() {
            self.sum_abs_delta += delta as f64;
        }
        if delta > self.max_abs_delta || self.worst_index.is_none() {
            self.max_abs_delta = delta;
            self.worst_index = Some(index.to_vec());
        }
    }

    pub(crate) fn report(&self, shape: &[usize], skip_examples: bool) -> String {
        let mut out = String::new();
        out.push_str("Got incorrect results: ");
        out.push_str(&format!(
            "{}/{} elements mismatched",
            self.mismatched, self.total
        ));

        if self.mismatched > 0 {
            let mean = if self.mismatched > 0 {
                self.sum_abs_delta / self.mismatched as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                " (max |Δ|={:.6}, mean |Δ|={:.6}",
                self.max_abs_delta, mean
            ));
            if let Some(idx) = &self.worst_index {
                out.push_str(&format!(", worst at {:?}", idx));
            }
            out.push(')');
            out.push_str(&format!(" — shape={:?}", shape));
        }

        // In Print modes, the per-element output is on stdout already; don't
        // re-list examples in the panic message.
        if skip_examples || self.mismatches.is_empty() {
            return out;
        }

        out.push_str("\nFirst mismatches:");
        for (idx, w) in &self.mismatches {
            out.push_str(&format!("\n  {:?}: {}", idx, format_wrong(w)));
        }
        if self.mismatched > self.mismatches.len() {
            out.push_str(&format!(
                "\n  ... and {} more",
                self.mismatched - self.mismatches.len()
            ));
        }

        out
    }
}

impl CompareVisitor for SummaryCollector {
    fn visit(&mut self, index: &[usize], status: ElemStatus) {
        self.total += 1;
        if let ElemStatus::Wrong(w) = status {
            self.mismatched += 1;
            let delta = match &w {
                WrongStatus::GotWrongValue { delta, .. } => *delta,
                WrongStatus::ExpectedNan { .. } | WrongStatus::GotNan { .. } => f32::INFINITY,
            };
            self.record_delta(index, delta);
            if self.mismatches.len() < self.max_reported {
                self.mismatches.push((index.to_vec(), w));
            }
        }
    }
}

/// Forwards each visit to a summary collector and (optionally) a printer, so
/// PrintAll/PrintFail keep their per-element stdout while the failure message
/// gets the aggregate stats too.
struct TeeVisitor<'a> {
    summary: &'a mut SummaryCollector,
    printer: Option<&'a mut dyn CompareVisitor>,
}

impl CompareVisitor for TeeVisitor<'_> {
    fn visit(&mut self, index: &[usize], status: ElemStatus) {
        // Clone status for the second visitor; ElemStatus is small (a few f32s).
        let status_for_summary = match &status {
            ElemStatus::Correct {
                got,
                delta,
                epsilon,
            } => ElemStatus::Correct {
                got: *got,
                delta: *delta,
                epsilon: *epsilon,
            },
            ElemStatus::Wrong(w) => ElemStatus::Wrong(clone_wrong(w)),
        };
        self.summary.visit(index, status_for_summary);
        if let Some(p) = self.printer.as_deref_mut() {
            p.visit(index, status);
        }
    }
}

fn clone_wrong(w: &WrongStatus) -> WrongStatus {
    match w {
        WrongStatus::GotWrongValue {
            got,
            expected,
            delta,
            epsilon,
        } => WrongStatus::GotWrongValue {
            got: *got,
            expected: *expected,
            delta: *delta,
            epsilon: *epsilon,
        },
        WrongStatus::ExpectedNan { got } => WrongStatus::ExpectedNan { got: *got },
        WrongStatus::GotNan { expected } => WrongStatus::GotNan {
            expected: *expected,
        },
    }
}

fn format_wrong(w: &WrongStatus) -> String {
    match w {
        WrongStatus::GotWrongValue {
            got,
            expected,
            delta,
            epsilon,
        } => format!(
            "got {got}, expected {expected}, |Δ|={delta} > ε={epsilon}",
        ),
        WrongStatus::ExpectedNan { got } => format!("got {got}, expected NaN"),
        WrongStatus::GotNan { expected } => format!("got NaN, expected {expected}"),
    }
}

#[inline]
fn compare_elem(got: f32, expected: f32, epsilon: f32) -> ElemStatus {
    let epsilon = (epsilon * expected).abs().max(epsilon);

    // NaN check: pass if both are NaN
    if got.is_nan() && expected.is_nan() {
        return ElemStatus::Correct {
            got,
            delta: 0.,
            epsilon,
        };
    }

    // NaN mismatch
    if got.is_nan() || expected.is_nan() {
        return if expected.is_nan() {
            ElemStatus::Wrong(WrongStatus::ExpectedNan { got })
        } else {
            ElemStatus::Wrong(WrongStatus::GotNan { expected })
        };
    }

    // Infinite check: pass if both inf with same sign
    if got.is_infinite() && expected.is_infinite() {
        if got.signum() == expected.signum() {
            return ElemStatus::Correct {
                got,
                delta: 0.,
                epsilon,
            };
        } else {
            return ElemStatus::Wrong(WrongStatus::GotWrongValue {
                got,
                expected,
                delta: f32::INFINITY,
                epsilon,
            });
        }
    }

    // Regular numeric comparison
    let delta = (got - expected).abs();
    if delta <= epsilon {
        ElemStatus::Correct {
            got,
            delta,
            epsilon,
        }
    } else {
        ElemStatus::Wrong(WrongStatus::GotWrongValue {
            got,
            expected,
            delta,
            epsilon,
        })
    }
}

fn compare_tensors(
    actual: &HostData,
    expected: &HostData,
    shape: &[usize],
    epsilon: f32,
    visitor: &mut dyn CompareVisitor,
    index: &mut Vec<usize>,
    slice: Option<&[std::ops::Range<usize>]>,
) -> bool {
    let mut failed = false;

    let dim = index.len();
    if dim == shape.len() {
        // Check if current index is within all ranges
        if let Some(slice) = slice {
            for (i, range) in index.iter().zip(slice.iter()) {
                if !range.contains(i) {
                    return false; // skip element outside slice
                }
            }
        }

        let got = actual.get_f32(index);
        let exp = expected.get_f32(index);
        let status = compare_elem(got, exp, epsilon);

        if matches!(status, ElemStatus::Wrong(_)) {
            failed = true;
        }

        visitor.visit(index, status);
        return failed;
    }

    // Recurse over full dimension — slice check happens at leaf
    for i in 0..shape[dim] {
        index.push(i);
        if compare_tensors(actual, expected, shape, epsilon, visitor, index, slice) {
            failed = true;
        }
        index.pop();
    }

    failed
}
