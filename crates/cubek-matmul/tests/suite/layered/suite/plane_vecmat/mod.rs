mod matmul_plane_vecmat {
    use crate::suite::layered::matmul_test_launcher::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Tma
    }

    #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_vecmat"))]
    mod vecmat {
        use super::*;
        use cubek_matmul::components::tile::io::Filled;
        pub type TMM =
            cubek_matmul::components::tile::plane_vec_mat_inner_product::PlaneVecMatInnerProduct<
                Filled,
            >;

        include!("algorithm.rs");
    }
}
