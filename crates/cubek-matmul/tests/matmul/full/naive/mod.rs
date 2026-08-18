mod f16_ty {
    use cubecl::frontend::Scalar;
    use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::elem_type_native()).as_global_elems()
    }

    include!("suite.rs");
}

mod f32_ty {
    use cubecl::frontend::Scalar;
    use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::elem_type_native()).as_global_elems()
    }

    include!("suite.rs");
}
