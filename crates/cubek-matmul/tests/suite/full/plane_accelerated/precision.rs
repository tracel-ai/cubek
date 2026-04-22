mod f16_ty {
    use super::*;
    use cubecl::frontend::CubePrimitive;
    use cubek_matmul::{definition::MatmulElems, definition::MatmulGlobalElems};

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    include!("tiling_scheme/tile.rs");
}

mod f32_ty {
    use super::*;
    use cubecl::frontend::CubePrimitive;
    use cubek_matmul::{definition::MatmulElems, definition::MatmulGlobalElems};

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("tiling_scheme/tile.rs");
}
