pub mod test_case;

macro_rules! testgen_reduce {
    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
        strategy: $strategy:expr,
    ) => {
        type TestDType = $dtype;
        fn test_shape() -> Vec<usize> {
            $shape
        }
        fn test_strides() -> Vec<usize> {
            $strides
        }
        fn test_axis() -> Option<usize> {
            $axis
        }

        fn test_strategy() -> ReduceStrategy {
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
    ) => {
        mod reduce_shared {
            type TestDType = $dtype;
            fn test_shape() -> Vec<usize> {
                $shape
            }
            fn test_strides() -> Vec<usize> {
                $strides
            }

            include!("reduce_shared.rs");
        }
    };

    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
    ) => {
        use cubek_reduce::{ReduceStrategy, routines::RoutineStrategy};

        mod full_cube {
            use super::*;
            use cubek_reduce::routines::cube::CubeStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy::FullCube(
                    RoutineStrategy::Strategy(CubeStrategy { use_planes: false })
                ),
            );
        }

        mod full_cube_plane {
            use super::*;
            use cubek_reduce::routines::cube::CubeStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy::FullCube(
                    RoutineStrategy::Strategy(CubeStrategy { use_planes: true })
                ),
            );
        }

        /// The goal of that test is to limit the size of a cube to `8` to validate multiple cubes
        /// arithmetic.
        ///
        /// With this test, we can't have `use_planes` to true since the `cube_dim.x !=
        /// plane_size`.
        mod full_cube_single_plane {
            use super::*;
            use cubek_reduce::routines::CubeReduceBlueprint;
            use cubek_reduce::BoundChecks;
            use cubecl::prelude::CubeDim;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy::FullCube(
                    RoutineStrategy::Forced(
                        CubeReduceBlueprint {
                            bound_checks: BoundChecks::Mask,
                            num_shared_accumulators: 8,
                            use_planes: false,
                        },
                        CubeDim::new_2d(8, 1),
                    )
                ),
            );
        }

        /// The goal of that test is to limit the size of a cube to `plane_size` to validate multiple planes
        /// arithmetic.
        mod full_plane_single_plane {
            use super::*;
            use cubek_reduce::routines::PlaneReduceBlueprint;
            use cubek_reduce::BoundChecks;
            use cubecl::prelude::CubeDim;

            mod plane_size_32 {
                use super::*;

                testgen_reduce!(
                    dtype: $dtype,
                    shape: $shape,
                    strides: $strides,
                    axis: $axis,
                    strategy: ReduceStrategy::FullPlane(
                        RoutineStrategy::Forced(
                            PlaneReduceBlueprint {
                                plane_idle: true,
                                bound_checks: BoundChecks::Mask,
                                independent: true,
                            },
                            CubeDim::new_2d(32, 2),
                        )
                    ),
                );
            }

            mod plane_size_64 {
                use super::*;

                testgen_reduce!(
                    dtype: $dtype,
                    shape: $shape,
                    strides: $strides,
                    axis: $axis,
                    strategy: ReduceStrategy::FullPlane(
                        RoutineStrategy::Forced(
                            PlaneReduceBlueprint {
                                plane_idle: true,
                                bound_checks: BoundChecks::Mask,
                                independent: true,
                            },
                            CubeDim::new_2d(64, 2),
                        )
                    ),
                );
            }
        }

        mod full_plane {
            use super::*;
            use cubek_reduce::routines::plane::PlaneStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy::FullPlane(
                    RoutineStrategy::Strategy(PlaneStrategy { independent: false })
                ),
            );
        }

        mod full_plane_unit {
            use super::*;
            use cubek_reduce::routines::plane::PlaneStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy::FullPlane(
                    RoutineStrategy::Strategy(PlaneStrategy { independent: true })
                ),
            );
        }

        mod full_unit {
            use super::*;
            use cubek_reduce::routines::unit::UnitStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy::FullUnit(RoutineStrategy::Strategy(UnitStrategy)),
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
        mod f16 {
            testgen_reduce!(
                dtype: half::f16,
                shape: $shape,
                strides: $strides,
                axis: $axis,
            );
        }
    };
    (
        shape: $shape:expr,
        strides: $strides:expr,
    ) => {
        mod f32 {
            testgen_reduce!(
                dtype: f32,
                shape: $shape,
                strides: $strides,
            );
        }
    };
}

mod reduce_dim {
    mod vector_small {
        testgen_reduce!(
            shape: vec![22],
            strides: vec![1],
            axis: Some(0),
        );
    }

    mod vector_large {
        testgen_reduce!(
            shape: vec![1024],
            strides: vec![1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_small {
        testgen_reduce!(
            shape: vec![4, 8],
            strides: vec![8, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_small {
        testgen_reduce!(
            shape: vec![4, 8],
            strides: vec![8, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_large {
        testgen_reduce!(
            shape: vec![8, 256],
            strides: vec![256, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_large {
        testgen_reduce!(
            shape: vec![8, 256],
            strides: vec![256, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_xlarge {
        testgen_reduce!(
            shape: vec![64, 1024],
            strides: vec![1024, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_xlarge {
        testgen_reduce!(
            shape: vec![64, 1024],
            strides: vec![1024, 1],
            axis: Some(0),
        );
    }

    mod parallel_rank_three_tensor {
        testgen_reduce!(
            shape: vec![16, 16, 16],
            strides: vec![1, 256, 16],
            axis: Some(0),
        );
    }

    mod perpendicular_rank_three_tensor {
        testgen_reduce!(
            shape: vec![16, 16, 16],
            strides: vec![1, 256, 16],
            axis: Some(1),
        );
    }

    mod parallel_rank_three_tensor_unexact_shape {
        testgen_reduce!(
            shape: vec![11, 12, 13],
            strides: vec![156, 13, 1],
            axis: Some(2),
        );
    }

    mod parallel_rank_four_tensor {
        testgen_reduce!(
            shape: vec![4, 4, 4, 4],
            strides: vec![1, 16, 64, 4],
            axis: Some(0),
        );
    }

    mod perpendicular_rank_four_tensor {
        testgen_reduce!(
            shape: vec![4, 4, 4, 4],
            strides: vec![1, 16, 64, 4],
            axis: Some(1),
        );
    }

    mod decreasing_rank_four_tensor {
        testgen_reduce!(
            shape: vec![4, 4, 4, 4],
            strides: vec![64, 16, 4, 1],
            axis: Some(3),
        );
    }

    mod parallel_matrix_with_jumps {
        testgen_reduce!(
            shape: vec![8, 8],
            strides: vec![64, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_with_jumps {
        testgen_reduce!(
            shape: vec![8, 8],
            strides: vec![64, 1],
            axis: Some(0),
        );
    }

    mod broadcast_slice_0 {
        testgen_reduce!(
            shape: vec![4, 32],
            strides: vec![0, 1],
            axis: Some(0),
        );
    }
}

mod reduce {
    mod vector_small {
        testgen_reduce!(
            shape: vec![22],
            strides: vec![1],
        );
    }

    mod vector_large {
        testgen_reduce!(
            shape: vec![1024],
            strides: vec![1],
        );
    }

    mod matrix_small {
        testgen_reduce!(
            shape: vec![4, 8],
            strides: vec![8, 1],
        );
    }

    mod matrix_large {
        testgen_reduce!(
            shape: vec![8, 256],
            strides: vec![256, 1],
        );
    }

    mod rank_three_tensor {
        testgen_reduce!(
            shape: vec![16, 16, 16],
            strides: vec![1, 256, 16],
        );
    }

    mod rank_three_tensor_unexact_shape {
        testgen_reduce!(
            shape: vec![11, 12, 13],
            strides: vec![156, 13, 1],
        );
    }

    mod rank_four_tensor {
        testgen_reduce!(
            shape: vec![4, 4, 4, 4],
            strides: vec![64, 16, 4, 1],
        );
    }

    // TODO: Doesn't work
    // mod matrix_with_jumps {
    //     testgen_reduce!(
    //         shape: vec![8, 8],
    //         strides: vec![64, 1],
    //     );
    // }

    mod broadcast_slice_0 {
        testgen_reduce!(
            shape: vec![4, 32],
            strides: vec![0, 1],
        );
    }
}
