mod matmul_plane_accelerated {
    use crate::suite::layered::matmul_test_launcher::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Normal
    }

    #[cfg(all(feature = "matmul_tests_plane", not(feature = "matmul_tests_mma")))]
    mod cmma {
        use super::*;
        use cubek_matmul::components::tile::io::Filled;
        pub type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

        include!("algorithm.rs");
    }

    #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_mma"))]
    mod mma {
        use super::*;
        type TMM = cubek_matmul::components::tile::mma::MmaMatmul;

        include!("algorithm.rs");
    }
}
