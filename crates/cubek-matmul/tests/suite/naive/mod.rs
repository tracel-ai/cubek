mod f16_ty {
    use cubek_matmul::tune_key::MatmulElemType;

    fn elem() -> MatmulElemType {
        MatmulElemType::new(half::f16::as_type_native_unchecked(), false)
    }

    include!("suite.rs");
}

mod f32_ty {
    use cubek_matmul::tune_key::MatmulElemType;

    fn elem() -> MatmulElemType {
        MatmulElemType::new(f32::as_type_native_unchecked(), false)
    }

    include!("suite.rs");
}
