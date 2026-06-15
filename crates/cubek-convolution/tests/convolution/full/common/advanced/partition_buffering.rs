#[macro_export]
macro_rules! testgen_convolution_partition_buffering {
    ($algorithm: expr, $dtypes: expr, $tiling_scheme: expr, $swizzle: expr) => {
        use cubek_matmul::components::stage::PartitionBuffering;

        $crate::testgen_convolution_problem!(
            $algorithm,
            $dtypes,
            $tiling_scheme,
            $swizzle,
            PartitionBuffering::Single
        );
    };
}
