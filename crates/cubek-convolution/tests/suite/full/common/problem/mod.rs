mod problem_size;

#[macro_export]
macro_rules! testgen_convolution_problem {
    ($algorithm: ty, $dtypes: expr, $tiling_scheme: expr, $swizzle: expr, $partition_buffering: expr) => {
        $crate::testgen_convolution_problem_size!(
            $algorithm,
            $dtypes,
            $tiling_scheme,
            $swizzle,
            $partition_buffering
        );
    };
}
