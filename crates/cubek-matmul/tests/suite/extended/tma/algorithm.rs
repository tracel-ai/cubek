#[cfg(feature = "matmul_tests_simple")]
mod simple_tma {
    use super::*;
    type Algorithm = cubek_matmul::routines::simple::SimpleTmaAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(feature = "matmul_tests_double")]
mod double_buffering_tma {
    use super::*;
    type Algorithm = cubek_matmul::routines::double_buffering::TmaDoubleBufferingAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(feature = "matmul_tests_double")]
mod specialized_tma {
    use super::*;
    type Algorithm = cubek_matmul::routines::specialized::SpecializedAlgorithm<TMM>;

    include!("precision.rs");
}
