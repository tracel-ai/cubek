#[macro_export]
macro_rules! testgen_convolution_accelerated_algorithm {
    () => {
        use cubek_convolution::ConvAlgorithm;

        mod simple_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SimpleSyncCyclic);
        }

        mod simple_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SimpleSyncStrided);
        }

        mod simple_tilewise {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SimpleSyncTilewise);
        }

        mod simple_async_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SimpleAsyncCyclic);
        }

        mod simple_async_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SimpleAsyncStrided);
        }

        mod simple_tma {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SimpleAsyncTma);
        }

        // Specialized async cyclic / strided are now wired through the
        // unified launch_ref but only the TMA variant is exercised here for
        // parity with the previous test surface.
        mod specialized_tma {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(ConvAlgorithm::SpecializedTma);
        }
    };
}
