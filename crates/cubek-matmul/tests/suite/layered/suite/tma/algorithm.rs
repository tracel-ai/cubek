#[cfg(feature = "matmul_tests_simple")]
mod simple_tma {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleTmaAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(feature = "matmul_tests_double")]
mod double_buffering_tma {
    use super::*;
    type Algorithm =
        cubek_matmul::kernels::layered::double_buffering::TmaDoubleBufferingAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(feature = "matmul_tests_double")]
mod specialized_tma {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::specialized::SpecializedAlgorithm<TMM>;

    include!("precision.rs");
}
