#[macro_export]
macro_rules! testgen_convolution_accelerated_precision {
    ($algorithm: expr) => {
        mod f16_ty {
            use super::*;
            use cubecl::prelude::Scalar;
            use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};

            fn dtypes() -> MatmulElems {
                let f16 = half::f16::elem_type_native();
                MatmulElems::from_globals(&MatmulGlobalElems {
                    lhs: f16,
                    rhs: f16,
                    out: f16,
                })
            }

            $crate::testgen_convolution_accelerated_tiling_scheme!($algorithm, dtypes());
        }

        mod f32_ty {
            use super::*;
            use cubecl::prelude::Scalar;
            use cubecl_common::tf32;
            use cubek_matmul::definition::MatmulElems;

            fn dtypes() -> MatmulElems {
                let f32 = f32::elem_type_native();
                let tf32 = tf32::elem_type_native();
                MatmulElems {
                    lhs_global: f32,
                    rhs_global: f32,
                    acc_global: f32,
                    lhs_stage: tf32,
                    rhs_stage: tf32,
                    acc_stage: f32,
                    lhs_register: tf32,
                    rhs_register: tf32,
                    acc_register: f32,
                }
            }

            $crate::testgen_convolution_accelerated_tiling_scheme!($algorithm, dtypes());
        }
    };
}
