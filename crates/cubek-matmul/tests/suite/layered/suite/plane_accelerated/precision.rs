#[cfg(feature = "matmul_tests_f16")]
mod f16_ty {
    use super::*;
    use cubecl::frontend::CubePrimitive;
    use cubek_matmul::components::MatmulElems;
    use cubek_matmul::tune_key::MatmulElemType;

    fn elems() -> MatmulElems {
        MatmulElems::from_single_dtype(MatmulElemType {
            dtype: half::f16::as_type_native_unchecked(),
            quantized: false,
        })
    }

    include!("tiling_scheme/tile.rs");
}

#[cfg(feature = "matmul_tests_f32")]
mod f32_ty {
    use super::*;
    use cubecl::frontend::CubePrimitive;
    use cubek_matmul::components::MatmulElems;
    use cubek_matmul::tune_key::MatmulElemType;

    fn elems() -> MatmulElems {
        MatmulElems::from_single_dtype(MatmulElemType {
            dtype: f32::as_type_native_unchecked(),
            quantized: false,
        })
    }

    include!("tiling_scheme/tile.rs");
}
