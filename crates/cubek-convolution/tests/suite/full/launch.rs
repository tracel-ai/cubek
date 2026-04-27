#[macro_export]
macro_rules! testgen_convolution_launch {
    ($algorithm: ty, $dtypes: expr, $tiling_scheme: expr, $swizzle: expr, $partition_buffering: expr, $problem_size: expr) => {
        use super::*;
        use $crate::suite::launcher_strategy::test_algo;

        #[test]
        pub fn test() {
            test_algo::<$algorithm>(
                $dtypes,
                $tiling_scheme,
                $swizzle,
                $partition_buffering,
                $problem_size,
            );
        }
    };
}
