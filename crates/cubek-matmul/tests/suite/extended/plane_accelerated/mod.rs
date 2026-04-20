mod matmul_plane_accelerated {
    use crate::suite::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Normal
    }

    #[cfg(all(feature = "matmul_tests_plane", not(feature = "matmul_tests_mma")))]
    mod cmma {
        use super::*;
        use cubek_std::tile::Strided;
        pub type TMM = cubek_matmul::components::tile_matmul::cmma::CmmaMatmul;

        include!("algorithm.rs");
    }

    #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_mma"))]
    mod mma {
        use super::*;
        type TMM = cubek_matmul::components::tile_matmul::mma::MmaMatmul;

        include!("algorithm.rs");
    }
}
