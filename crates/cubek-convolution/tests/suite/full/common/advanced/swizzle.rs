#[macro_export]
macro_rules! testgen_convolution_swizzle {
    ($algorithm: ty, $dtypes: expr, $tiling_scheme: expr) => {
        use cubek_matmul::definition::SwizzleModes;
        use cubek_std::stage::SwizzleMode;

        mod none {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $dtypes,
                $tiling_scheme,
                SwizzleModes {
                    lhs: SwizzleMode::None,
                    rhs: SwizzleMode::None,
                    ..Default::default()
                }
            );
        }

        mod b32 {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $dtypes,
                $tiling_scheme,
                SwizzleModes {
                    lhs: SwizzleMode::B32,
                    rhs: SwizzleMode::B32,
                    ..Default::default()
                }
            );
        }

        mod b64 {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $dtypes,
                $tiling_scheme,
                SwizzleModes {
                    lhs: SwizzleMode::B64,
                    rhs: SwizzleMode::B64,
                    ..Default::default()
                }
            );
        }

        mod b128 {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $dtypes,
                $tiling_scheme,
                SwizzleModes {
                    lhs: SwizzleMode::B128,
                    rhs: SwizzleMode::B128,
                    ..Default::default()
                }
            );
        }
    };
}
