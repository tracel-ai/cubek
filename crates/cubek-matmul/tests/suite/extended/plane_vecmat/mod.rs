mod matmul_plane_vecmat {
    use crate::suite::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Normal
    }

    #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_vecmat"))]
    mod vecmat {
        use super::*;
        use cubek_std::tile::Strided;
        pub type TMM =
            cubek_matmul::components::tile_matmul::plane_vec_mat_inner_product::PlaneVecMatInnerProduct;

        include!("algorithm.rs");
    }
}
