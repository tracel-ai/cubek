//! The `reduce_dim` matrix below is a cross product of shapes, dtypes and the kernels a strategy
//! names, so its size multiplies fast. Tests with no `cfg` are the light subset the base suite
//! runs, `#[cfg(feature = "heavy")]` adds the cube kernel over that same subset, and
//! `#[cfg(feature = "extended")]` restores the whole cross product.
//!
//! The light subset keeps one point per collapsed axis and one shape per layout class. A shape
//! earns its own light slot when it changes how the reduce axis sits in memory, not when it only
//! changes how many cubes and planes the same layout needs.

macro_rules! testgen_reduce {
    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
        strategy: $strategy:expr,
    ) => {
        use cubecl::zspace::{Shape, Strides, shape, strides};
        use cubek_reduce::kernels::ReduceOpStrategy;

        type TestDType = $dtype;

        fn test_shape() -> Shape {
            $shape
        }
        fn test_strides() -> Strides {
            $strides
        }
        fn test_axis() -> Option<usize> {
            $axis
        }

        fn test_strategy() -> ReduceOpStrategy {
            $strategy
        }

        mod reduce_dim {
            use super::*;
            include!("reduce_dim.rs");
        }
    };

    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
    ) => {
        /// The cube kernel is expensive on CPU and can stall CI, so it is excluded from the light
        /// test suite.
        #[cfg(feature = "heavy")]
        mod cube {
            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: crate::reduce::it::test_case::CUBE,
            );
        }

        mod plane {
            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: crate::reduce::it::test_case::PLANE,
            );
        }

        mod unit {
            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: crate::reduce::it::test_case::UNIT,
            );
        }

        mod rule {
            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: crate::reduce::it::test_case::RULE,
            );
        }
    };

    (
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
    ) => {
        mod f32 {
            testgen_reduce!(
                dtype: f32,
                shape: $shape,
                strides: $strides,
                axis: $axis,
            );
        }
        #[cfg(feature = "extended")]
        mod f16 {
            testgen_reduce!(
                dtype: half::f16,
                shape: $shape,
                strides: $strides,
                axis: $axis,
            );
        }
    };
}

mod reduce_dim {
    mod vector_small {
        testgen_reduce!(
            shape: shape![22],
            strides: strides![1],
            axis: Some(0),
        );
    }

    #[cfg(feature = "extended")]
    mod vector_large {
        testgen_reduce!(
            shape: shape![1024],
            strides: strides![1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_small {
        testgen_reduce!(
            shape: shape![4, 8],
            strides: strides![8, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_small {
        testgen_reduce!(
            shape: shape![4, 8],
            strides: strides![8, 1],
            axis: Some(0),
        );
    }

    /// A row axis one long after the reduced axis: the lines still lie along the reduced axis.
    mod parallel_before_an_axis_one_long {
        testgen_reduce!(
            shape: shape![128, 32, 1],
            strides: strides![32, 1, 1],
            axis: Some(1),
        );
    }

    // `*_large_odd_batch` below already reduces an axis long enough to need
    // more than one cube and plane, so `large` through `xxlarge` only scale
    // that same count.
    #[cfg(feature = "extended")]
    mod parallel_matrix_large {
        testgen_reduce!(
            shape: shape![8, 256],
            strides: strides![256, 1],
            axis: Some(1),
        );
    }

    #[cfg(feature = "extended")]
    mod perpendicular_matrix_large {
        testgen_reduce!(
            shape: shape![8, 256],
            strides: strides![256, 1],
            axis: Some(0),
        );
    }

    #[cfg(feature = "extended")]
    mod parallel_matrix_xlarge {
        testgen_reduce!(
            shape: shape![64, 1024],
            strides: strides![1024, 1],
            axis: Some(1),
        );
    }

    #[cfg(feature = "extended")]
    mod perpendicular_matrix_xlarge {
        testgen_reduce!(
            shape: shape![64, 1024],
            strides: strides![1024, 1],
            axis: Some(0),
        );
    }

    #[cfg(feature = "extended")]
    mod parallel_matrix_xxlarge {
        testgen_reduce!(
            shape: shape![64*4, 1024*4],
            strides: strides![1024*4, 1],
            axis: Some(1),
        );
    }

    #[cfg(feature = "extended")]
    mod perpendicular_matrix_xxlarge {
        testgen_reduce!(
            shape: shape![64*4, 1024*4],
            strides: strides![1024*4, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_large_odd_batch {
        testgen_reduce!(
            shape: shape![33, 1024],
            strides: strides![1024, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_large_odd_batch {
        testgen_reduce!(
            shape: shape![33, 1024],
            strides: strides![1024, 1],
            axis: Some(0),
        );
    }

    // The only light shape whose reduce axis is contiguous without being the
    // innermost one, which is where parallel vectorization has to read a
    // permuted layout rather than a row.
    mod parallel_rank_three_tensor {
        testgen_reduce!(
            shape: shape![16, 16, 16],
            strides: strides![1, 256, 16],
            axis: Some(0),
        );
    }

    mod perpendicular_rank_three_tensor {
        testgen_reduce!(
            shape: shape![16, 16, 16],
            strides: strides![1, 256, 16],
            axis: Some(1),
        );
    }

    mod parallel_rank_three_tensor_unexact_shape {
        testgen_reduce!(
            shape: shape![11, 12, 13],
            strides: strides![156, 13, 1],
            axis: Some(2),
        );
    }

    // The permuted layout these two reduce is already light at rank three, and
    // `decreasing_rank_four_tensor` covers rank four contiguously.
    #[cfg(feature = "extended")]
    mod parallel_rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![1, 16, 64, 4],
            axis: Some(0),
        );
    }

    #[cfg(feature = "extended")]
    mod perpendicular_rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![1, 16, 64, 4],
            axis: Some(1),
        );
    }

    mod decreasing_rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![64, 16, 4, 1],
            axis: Some(3),
        );
    }

    // Padded rows leave gaps between the reductions, so this is the only light
    // shape where a vectorized read of the reduce axis must stop at the row
    // rather than at the end of the allocation.
    mod parallel_matrix_with_jumps {
        testgen_reduce!(
            shape: shape![256, 256],
            strides: strides![512, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_with_jumps {
        testgen_reduce!(
            shape: shape![256, 256],
            strides: strides![512, 1],
            axis: Some(0),
        );
    }

    mod broadcast_slice_0 {
        testgen_reduce!(
            shape: shape![4, 32],
            strides: strides![0, 1],
            axis: Some(0),
        );
    }

    // One small point for the axis the light suite collapses, so that `f16` stays covered. Under
    // `extended` the full cross product already contains it.
    #[cfg(not(feature = "extended"))]
    mod collapsed_axis_coverage {
        mod f16 {
            testgen_reduce!(
                dtype: half::f16,
                shape: shape![4, 8],
                strides: strides![8, 1],
                axis: Some(1),
            );
        }
    }
}
