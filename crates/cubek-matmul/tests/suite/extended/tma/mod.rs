mod matmul_tma {
    use crate::suite::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Tma
    }

    #[cfg(all(feature = "matmul_tests_tma", not(feature = "matmul_tests_mma")))]
    mod cmma {
        use super::*;
        pub type TMM = cubek_matmul::components::tile_matmul::cmma::CmmaMatmul;

        include!("algorithm.rs");
    }

    #[cfg(all(feature = "matmul_tests_tma", feature = "matmul_tests_mma"))]
    mod mma {
        use super::*;
        pub type TMM = cubek_matmul::components::tile_matmul::mma::MmaMatmul;

        include!("algorithm.rs");
    }
}
