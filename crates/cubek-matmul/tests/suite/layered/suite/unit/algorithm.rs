#[cfg(feature = "matmul_tests_simple")]
mod simple {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple_unit::SimpleUnitAlgorithm;

    include!("precision.rs");
}

#[cfg(feature = "matmul_tests_double")]
mod double_buffering {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::double_unit::DoubleUnitAlgorithm;

    include!("precision.rs");
}
