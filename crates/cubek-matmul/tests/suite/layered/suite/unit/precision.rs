#[cfg(feature = "matmul_tests_f16")]
mod f16_ty {
    use super::*;
    use cubecl::frontend::CubePrimitive;
    use cubek_matmul::definition::MatmulElems;
    use cubek_matmul::definition::MatmulGlobalElems;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    include!("tiling_scheme/tile.rs");
}

#[cfg(feature = "matmul_tests_f32")]
mod f32_ty {
    use super::*;
    use cubecl::frontend::CubePrimitive;
    use cubek_matmul::definition::MatmulElems;
    use cubek_matmul::definition::MatmulGlobalElems;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("tiling_scheme/tile.rs");
}
