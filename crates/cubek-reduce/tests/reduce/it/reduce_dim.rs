use crate::reduce::it::test_case::TestCase;

#[test]
pub fn test_argmax() {
    test_case().test_argmax();
}

#[test]
pub fn test_argmin() {
    test_case().test_argmin();
}

// `k` only changes how many accumulator slices the routine keeps, which does
// not interact with the layout this matrix varies, so the light suite runs one
// `k` per operation and `extended` restores the sweep.
#[test]
pub fn test_argtopk_3() {
    test_case().test_argtopk(3);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_argtopk_5() {
    test_case().test_argtopk(5);
}

#[test]
pub fn test_topk_3() {
    test_case().test_topk(3);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_topk_5() {
    test_case().test_topk(5);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_topk_with_indices_1() {
    test_case().test_topk_with_indices(1);
}

#[test]
pub fn test_topk_with_indices_3() {
    test_case().test_topk_with_indices(3);
}

#[cfg(feature = "extended")]
#[test]
pub fn test_topk_with_indices_5() {
    test_case().test_topk_with_indices(5);
}

#[test]
pub fn test_min_with_indices() {
    test_case().test_min_with_indices();
}

#[test]
pub fn test_max_with_indices() {
    test_case().test_max_with_indices();
}

#[test]
pub fn test_mean() {
    test_case().test_mean();
}

#[test]
pub fn test_sum() {
    test_case().test_sum();
}

#[test]
pub fn test_prod() {
    test_case().test_prod();
}

#[test]
pub fn test_min() {
    test_case().test_min();
}

#[test]
pub fn test_max() {
    test_case().test_max();
}

#[test]
pub fn test_max_abs() {
    test_case().test_max_abs();
}

#[test]
pub fn test_any() {
    test_case().test_any();
}

#[test]
pub fn test_all() {
    test_case().test_all();
}

fn test_case() -> TestCase {
    TestCase::new::<TestDType>(test_shape(), test_strides(), test_axis(), test_strategy())
}
