mod matmul_tma {
    use crate::suite::layered::matmul_test_launcher::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Tma
    }

    #[cfg(all(feature = "matmul_tests_tma", not(feature = "matmul_tests_mma")))]
    mod cmma {
        use super::*;
        use cubek_matmul::components::tile::io::Filled;
        pub type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

        include!("algorithm.rs");
    }

    #[cfg(all(feature = "matmul_tests_tma", feature = "matmul_tests_mma"))]
    mod mma {
        use super::*;
        pub type TMM = cubek_matmul::components::tile::mma::MmaMatmul;

        include!("algorithm.rs");
    }
}
