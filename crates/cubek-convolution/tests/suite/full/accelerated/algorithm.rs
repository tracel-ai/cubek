#[macro_export]
macro_rules! testgen_convolution_accelerated_algorithm {
    () => {
        use cubek_convolution::{
            kernels::algorithm::simple::*, kernels::algorithm::specialized::*,
        };

        mod simple_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleSyncCyclicConv<TMM>);
        }

        mod simple_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleSyncStridedConv<TMM>);
        }

        mod simple_tilewise {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleSyncTilewiseConv<TMM>);
        }

        mod simple_async_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleAsyncCyclicConv<TMM>);
        }

        mod simple_async_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleAsyncStridedConv<TMM>);
        }

        mod simple_tma {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleAsyncTmaConv<TMM>);
        }

        // Specialized async cyclic / strided are currently broken; only the TMA
        // variant is wired up.
        mod specialized_tma {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SpecializedTmaConv<TMM>);
        }
    };
}
