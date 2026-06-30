mod cgr;
pub mod house;
mod mgs;
pub mod solve;

#[macro_export]
macro_rules! testgen_qr_cgr {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::qr::cgr::test_qr_cgr::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::qr::cgr::test_qr_cgr::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::qr::cgr::test_qr_cgr::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::qr::cgr::test_qr_cgr::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::qr::cgr::test_qr_cgr_rect::<FloatT>(517, 157);
            }

    };
    ([$($float:ident),*]) => {
        mod cgr {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_cgr!($float);
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_qr_house {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::qr::house::test_qr_baht::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::qr::house::test_qr_baht::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::qr::house::test_qr_baht::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::qr::house::test_qr_baht::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::qr::house::test_qr_baht_rect::<FloatT>(517, 157);
            }

    };
    ([$($float:ident),*]) => {
        mod house {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_house!($float);
                })*
            }
        }
    };
}

mod test_cgr {
    testgen_qr_cgr!([f32, f64]);
}

mod test_house {
    testgen_qr_house!([f32, f64]);
}

#[macro_export]
macro_rules! testgen_qr_mgs {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::qr::mgs::test_qr_mgs::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::qr::mgs::test_qr_mgs::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::qr::mgs::test_qr_mgs::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::qr::mgs::test_qr_mgs::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::qr::mgs::test_qr_mgs_rect::<FloatT>(517, 157);
            }

    };
    ([$($float:ident),*]) => {
        mod mgs {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_mgs!($float);
                })*
            }
        }
    };
}

mod test_mgs {
    testgen_qr_mgs!([f32, f64]);
}


#[macro_export]
macro_rules! testgen_solve {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_solve_square_house() {
                crate::suite::qr::solve::test_solve_square::<FloatT>(16, cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder);
            }

            #[test]
            pub fn test_solve_square_givens() {
                crate::suite::qr::solve::test_solve_square::<FloatT>(16, cubek_linalg::QRStrategy::CommonGivensRotations);
            }

            #[test]
            pub fn test_solve_rect_house() {
                crate::suite::qr::solve::test_solve_rect::<FloatT>(32, 16, cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder);
            }

            #[test]
            pub fn test_solve_rect_givens() {
                crate::suite::qr::solve::test_solve_rect::<FloatT>(32, 16, cubek_linalg::QRStrategy::CommonGivensRotations);
            }

            #[test]
            pub fn test_solve_square_mgs() {
                crate::suite::qr::solve::test_solve_square::<FloatT>(16, cubek_linalg::QRStrategy::ModifiedGramSchmidt);
            }

            #[test]
            pub fn test_solve_rect_mgs() {
                crate::suite::qr::solve::test_solve_rect::<FloatT>(32, 16, cubek_linalg::QRStrategy::ModifiedGramSchmidt);
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

#[macro_export]
macro_rules! testgen_qr_tsqr {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::qr::house::test_qr_tsqr::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::qr::house::test_qr_tsqr::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::qr::house::test_qr_tsqr::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::qr::house::test_qr_tsqr::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::qr::house::test_qr_tsqr_rect::<FloatT>(517, 157);
            }

    };
    ([$($float:ident),*]) => {
        mod tsqr {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_tsqr!($float);
                })*
            }
        }
    };
}

mod test_tsqr {
    testgen_qr_tsqr!([f32, f64]);
}
