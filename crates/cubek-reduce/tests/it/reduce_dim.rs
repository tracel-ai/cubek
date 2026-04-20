use crate::it::test_case::TestCase;

#[test]
pub fn test_argmax() {
    test_case().test_argmax();
}

#[test]
pub fn test_argmin() {
    test_case().test_argmin();
}

#[test]
#[ignore = "Arg Top k not yet implemented"]
pub fn test_argtopk() {
    test_case().test_argtopk(3);
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

fn test_case() -> TestCase {
    TestCase::new::<TestDType>(test_shape(), test_strides(), test_axis(), test_strategy())
}
