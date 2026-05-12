mod normal {
    mod f32 {
        type TestDType = f32;
        include!("normal.rs");
    }
    mod f16 {
        type TestDType = half::f16;
        include!("normal.rs");
    }
}

mod bernoulli {
    type TestDType = f32;
    include!("bernoulli.rs");
}

mod interval {
    include!("interval.rs");
}

mod uniform {
    type TestDType = f32;

    include!("uniform.rs");
}
