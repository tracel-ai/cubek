mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_convolution_accelerated {
    () => {
        mod conv2d_accelerated {
            use super::*;

            mod cmma {
                use super::*;
                $crate::testgen_convolution_accelerated_algorithm!();
            }

            mod mma {
                use super::*;
                $crate::testgen_convolution_accelerated_algorithm!();
            }
        }
    };
}
