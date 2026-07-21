pub mod baht_tsqr;
pub mod errors;
pub mod solve;

#[macro_export]
macro_rules! testgen_qr_baht_tsqr {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::qr::baht_tsqr::test_qr::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::qr::baht_tsqr::test_qr::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::qr::baht_tsqr::test_qr::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::qr::baht_tsqr::test_qr::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::qr::baht_tsqr::test_qr_rect::<FloatT>(517, 157);
            }

            #[test]
            pub fn test_rect_row_major() {
                crate::suite::qr::baht_tsqr::test_qr_rect_row_major::<FloatT>(157, 47);
            }

            #[test]
            pub fn test_tf32_opt_in() {
                crate::suite::qr::baht_tsqr::test_qr_tf32::<FloatT>(157);
            }

    };
    ([$($float:ident),*]) => {
        mod baht_tsqr {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_baht_tsqr!($float);
                })*
            }
        }
    };
}

mod test_baht_tsqr {
    testgen_qr_baht_tsqr!([f32, f64]);
}

#[macro_export]
macro_rules! testgen_solve {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_solve_square() {
                crate::suite::qr::solve::test_solve_square::<FloatT>(16);
            }

            #[test]
            pub fn test_solve_rect() {
                crate::suite::qr::solve::test_solve_rect::<FloatT>(32, 16);
            }
    };
    ([$($float:ident),*]) => {
        mod solve {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_solve!($float);
                })*
            }
        }
    };
}

mod test_solve {
    testgen_solve!([f32, f64]);
}
