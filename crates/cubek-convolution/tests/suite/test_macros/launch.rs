#[macro_export]
macro_rules! testgen_convolution_launch {
    ($algorithm: ty, $precision: ty, $tiling_scheme: expr, $swizzle: expr, $partition_buffering: expr, $problem_size: expr) => {
        use super::*;
        use $crate::suite::test_macros::suite::test_algo;

        #[test]
        pub fn test() {
            test_algo::<$algorithm, $precision>(
                $tiling_scheme,
                $swizzle,
                $partition_buffering,
                $problem_size,
            );
        }
    };
}
