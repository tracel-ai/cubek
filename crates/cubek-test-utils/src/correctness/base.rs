use crate::correctness::color_printer::ColorPrinter;
use crate::test_mode::{TestMode, current_test_mode};
use crate::{HostData, ValidationResult};

pub fn assert_equals_approx(
    actual: &HostData,
    expected: &HostData,
    epsilon: f32,
) -> ValidationResult {
    if actual.shape != expected.shape {
        return ValidationResult::Fail(format!(
            "Shape mismatch: got {:?}, expected {:?}",
            actual.shape, expected.shape,
        ));
    }

    let shape = &actual.shape;
    let test_mode = current_test_mode();

    let mut visitor: Box<dyn CompareVisitor> = match test_mode.clone() {
        TestMode::Print {
            filter,
            fail_only: _,
        } => {
            if !filter.is_empty() && filter.len() != shape.len() {
                return ValidationResult::Skipped(format!(
                    "Print mode activated with invalid filter rank. Got {:?}, expected {:?}",
                    filter.len(),
                    shape.len()
                ));
            }
            Box::new(ColorPrinter::new(filter))
        }
        _ => Box::new(FailFast),
    };

    let test_failed = compare_tensors(
        actual,
        expected,
        shape,
        epsilon,
        &mut *visitor,
        &mut Vec::new(),
    );

    match test_failed {
        true => ValidationResult::Fail("Got incorrect results".to_string()),
        false => ValidationResult::Pass,
    }
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
    let eps = (epsilon * expected).abs().max(epsilon).min(0.99);

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
) -> bool {
    let mut failed = false;

    let dim = index.len();
    if dim == shape.len() {
        let got = actual.get_f32(index);
        let exp = expected.get_f32(index);

        let status = compare_elem(got, exp, epsilon);
        if matches!(status, ElemStatus::Wrong(_)) {
            failed = true;
        }
        visitor.visit(index, status);
        return failed;
    }

    for i in 0..shape[dim] {
        index.push(i);
        if compare_tensors(actual, expected, shape, epsilon, visitor, index) {
            failed = true;
        }
        index.pop();
    }

    failed
}
