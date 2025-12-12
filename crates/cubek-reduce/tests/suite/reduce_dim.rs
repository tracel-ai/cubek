use crate::suite::test_case::TestCase;

#[test]
pub fn test_argmax() {
    test_case().test_argmax();
}

#[test]
pub fn test_argmin() {
    test_case().test_argmin();
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

fn test_case() -> TestCase<TestDType> {
    TestCase::<TestDType> {
        shape: test_shape(),
        stride: test_strides(),
        axis: test_axis(),
        strategy: test_strategy(),
        elem: core::marker::PhantomData,
    }
}
