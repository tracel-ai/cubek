mod f16_ty {
    use cubek_matmul::definition::MatmulElemType;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(MatmulElemType::new(
            half::f16::as_type_native_unchecked(),
            false,
        ))
        .as_global_elems()
    }

    include!("suite.rs");
}

mod f32_ty {
    use cubek_matmul::definition::MatmulElemType;

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(MatmulElemType::new(f32::as_type_native_unchecked(), false))
            .as_global_elems()
    }

    include!("suite.rs");
}
