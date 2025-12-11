#[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_cyclic"))]
mod simple_cyclic {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_cyclic"))]
mod double_buffering_cyclic {
    use super::*;
    type Algorithm =
        cubek_matmul::kernels::layered::double_buffering::CyclicDoubleBufferingAlgorithm<TMM>;

    include!("precision.rs");
}
