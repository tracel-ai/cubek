use std::ops::Range;

use crate::correctness::color_printer::ColorPrinter;
use crate::test_mode::{TestMode, current_test_mode};
use crate::{HostData, ValidationResult};

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

    let mut visitor: Box<dyn CompareVisitor> = match test_mode.clone() {
        TestMode::Print { filter, .. } => Box::new(SafePrinter {
            inner: ColorPrinter::new(filter),
            shape_len: shape.len(),
        }),
        _ => Box::new(FailFast),
    };

    let test_failed = compare_tensors(
        lhs,
        rhs,
        shape,
        epsilon,
        &mut *visitor,
        &mut Vec::new(),
        slice.as_deref(), // pass slice as Option<&[usize]>
    );

    // Enforce filter rank only if the test failed and we would print
    if test_failed {
        if let TestMode::Print { filter, .. } = test_mode
            && !filter.is_empty()
            && filter.len() != shape.len()
        {
            return ValidationResult::Skipped(format!(
                "Print mode activated with invalid filter rank. Got {:?}, expected {:?}",
                filter.len(),
                shape.len()
            ));
        }

        return ValidationResult::Fail("Got incorrect results".to_string());
    }

    ValidationResult::Pass
}

#[derive(Debug)]
pub(crate) enum ElemStatus {
    Correct { got: f32 },
    Wrong(WrongStatus),
}

#[derive(Debug)]
pub(crate) enum WrongStatus {
    GotWrongValue {
        got: f32,
        expected: f32,
        diff: f32,
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

pub(crate) struct FailFast;

impl CompareVisitor for FailFast {
    fn visit(&mut self, index: &[usize], status: ElemStatus) {
        if let ElemStatus::Wrong(w) = status {
            panic!("Mismatch at {:?}: {:?}", index, w);
        }
    }
}

#[inline]
fn compare_elem(got: f32, expected: f32, epsilon: f32) -> ElemStatus {
    let eps = (epsilon * expected).abs().max(epsilon);

    // NaN check: pass if both are NaN
    if got.is_nan() && expected.is_nan() {
        return ElemStatus::Correct { got };
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
            return ElemStatus::Correct { got };
        } else {
            return ElemStatus::Wrong(WrongStatus::GotWrongValue {
                got,
                expected,
                diff: f32::INFINITY,
                epsilon: eps,
            });
        }
    }

    // Regular numeric comparison
    let diff = (got - expected).abs();
    if diff < eps {
        ElemStatus::Correct { got }
    } else {
        ElemStatus::Wrong(WrongStatus::GotWrongValue {
            got,
            expected,
            diff,
            epsilon: eps,
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
