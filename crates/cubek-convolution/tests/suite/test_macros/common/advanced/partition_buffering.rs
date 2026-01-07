#[macro_export]
macro_rules! testgen_convolution_partition_buffering {
    ($algorithm: ty, $precision: ty, $tiling_scheme: expr, $swizzle: expr) => {
        use cubek_matmul::components::stage::PartitionBuffering;

        #[cfg(not(feature = "conv_tests_partition_buffering"))]
        $crate::testgen_convolution_problem!(
            $algorithm,
            $precision,
            $tiling_scheme,
            $swizzle,
            PartitionBuffering::Single
        );

        #[cfg(feature = "conv_tests_partition_buffering")]
        mod pb1 {
            use super::*;

            $crate::testgen_convolution_problem!(
                $algorithm,
                $precision,
                $tiling_scheme,
                $swizzle,
                PartitionBuffering::Single
            );
        }

        #[cfg(feature = "conv_tests_partition_buffering")]
        mod pb2 {
            use super::*;

            $crate::testgen_convolution_problem!(
                $algorithm,
                $precision,
                $tiling_scheme,
                $swizzle,
                PartitionBuffering::Double
            );
        }
    };
}
