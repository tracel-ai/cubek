use crate::test_utils::correctness::color_printer::ColorPrinter;
use crate::test_utils::test_mode::{TestMode, current_test_mode};
use crate::test_utils::test_tensor::copy_casted;
use cubecl::CubeElement;
use cubecl::frontend::CubePrimitive;
use cubecl::{TestRuntime, client::ComputeClient, std::tensor::TensorHandle};

pub fn assert_equals_approx(
    client: &ComputeClient<TestRuntime>,
    handle: &TensorHandle<TestRuntime>,
    expected: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    let data_handle = copy_casted(client, handle, f32::as_type_native_unchecked());
    let actual =
        f32::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();
    let shape = handle.shape.clone();

    let test_mode = current_test_mode();

    let mut visitor: Box<dyn CompareVisitor> = match test_mode.clone() {
        TestMode::Print {
            filter,
            only_failing: _,
        } => {
            if filter.len() > 0 && filter.len() != shape.len() {
                return Err(format!(
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
        &actual,
        &expected,
        &shape,
        epsilon,
        &mut *visitor,
        &mut Vec::new(),
    );

    match test_mode {
        TestMode::Print { only_failing, .. } => {
            if !only_failing || test_failed {
                Err("Print mode activated".to_string())
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
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
    let eps = (epsilon * expected).abs().max(epsilon);

    let actual_nan = got.is_nan();
    let expected_nan = expected.is_nan();

    if actual_nan != expected_nan {
        if expected_nan {
            return ElemStatus::Wrong(WrongStatus::ExpectedNan { got });
        } else {
            return ElemStatus::Wrong(WrongStatus::GotNan { expected });
        }
    }

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
    actual_values: &[f32],
    expected_values: &[f32],
    shape: &[usize],
    epsilon: f32,
    visitor: &mut dyn CompareVisitor,
    index: &mut Vec<usize>,
) -> bool {
    let mut failed = false;
    if shape.len() == 1 {
        for i in 0..shape[0] {
            index.push(i);
            let got = actual_values[i];
            let expected = expected_values[i];
            let status = compare_elem(got, expected, epsilon);
            if matches!(status, ElemStatus::Wrong(_)) {
                failed = true;
            }
            visitor.visit(index, status);
            index.pop();
        }
    } else {
        let stride: usize = shape[1..].iter().product();
        for i in 0..shape[0] {
            index.push(i);
            if compare_tensors(
                &actual_values[i * stride..(i + 1) * stride],
                &expected_values[i * stride..(i + 1) * stride],
                &shape[1..],
                epsilon,
                visitor,
                index,
            ) {
                failed = true;
            }
            index.pop();
        }
    }
    failed
}
