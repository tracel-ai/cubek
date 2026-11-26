use crate::reduce::it::test_case::{Op, ReduceScopeChecker, TestCase};
use cubek_test_utils::ScopeChecker;

#[test]
pub fn test_argmax() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::ArgMax)]);
}

#[test]
pub fn test_argmin() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::ArgMin)]);
}

// `k` only changes how many accumulator slices the routine keeps, which does
// not interact with the layout this matrix varies, so the light suite runs one
// `k` per operation and `extended` restores the sweep, up to the shortest row of the matrix.
#[test]
pub fn test_argtopk_3() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::ArgTopK(3))]);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_argtopk_4() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::ArgTopK(4))]);
}

#[test]
pub fn test_topk_3() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::TopK(3))]);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_topk_4() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::TopK(4))]);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_topk_with_indices_1() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::TopKWithIndices(1))]);
}

#[test]
pub fn test_topk_with_indices_3() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::TopKWithIndices(3))]);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_topk_with_indices_4() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::TopKWithIndices(4))]);
}

#[test]
pub fn test_min_with_indices() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::MinWithIndices)]);
}

#[test]
pub fn test_max_with_indices() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::MaxWithIndices)]);
}

#[test]
pub fn test_mean() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::Mean)]);
}

#[test]
pub fn test_sum() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::Sum)]);
}

#[test]
pub fn test_prod() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::Prod)]);
}

#[test]
pub fn test_min() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::Min)]);
}

#[test]
pub fn test_max() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::Max)]);
}

#[test]
pub fn test_max_abs() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::MaxAbs)]);
}

#[test]
pub fn test_any() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::Any)]);
}

#[test]
pub fn test_all() {
    ReduceScopeChecker.check([test_case().reduced_by(Op::All)]);
}

fn test_case() -> TestCase {
    TestCase::new::<TestDType>(test_shape(), test_strides(), test_axis(), test_strategy())
}
