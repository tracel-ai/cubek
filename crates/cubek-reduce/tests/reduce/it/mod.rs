//! The `reduce_dim` matrix below is a cross product of shape configurations,
//! dtypes, output-vectorization settings, forced plane dims and routines, so
//! its size multiplies fast. Tests with no `cfg` are the light subset the base
//! suite runs, `#[cfg(feature = "heavy")]` adds the `Cube` routines over that
//! same subset, and `#[cfg(feature = "extended")]` restores the whole cross
//! product.
//!
//! The light subset keeps one point per collapsed axis and one shape per
//! layout class. A shape earns its own light slot when it changes how the
//! reduce axis sits in memory, not when it only changes how many cubes and
//! planes the same layout needs.

mod test_case;

mod argtopk_shared_memory;
mod logical;
mod nan_extrema;
mod plane_reduction;
mod topk_with_indices_cube;
mod with_indices_validation;

macro_rules! testgen_reduce {
    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
        strategy: $strategy:expr,
    ) => {
        use cubecl::zspace::{Shape, Strides, shape, strides};

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
            use cubecl::zspace::{Shape, Strides, shape, strides};

            type TestDType = $dtype;

            fn test_shape() -> Shape {
                $shape
            }
            fn test_strides() -> Strides {
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
        mod parallel_vectorization_enabled {
            use cubek_reduce::launch::VectorizationStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                vectorization_strategy: VectorizationStrategy {
                    parallel_output_vectorization: true,
                },
            );
        }

        #[cfg(feature = "extended")]
        mod parallel_vectorization_disabled {
            use cubek_reduce::launch::VectorizationStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                vectorization_strategy: VectorizationStrategy {
                    parallel_output_vectorization: false,
                },
            );
        }
    };

    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
        vectorization_strategy: $vectorization_strategy:expr,
    ) => {
        use cubek_reduce::{ReduceStrategy, routines::BlueprintStrategy, launch::RoutineStrategy};
        use cubek_reduce::routines::PlaneMergeStrategy;
        use cubecl::config::autotune::AutotuneLevel;

        /// Cube-routine tests are expensive on CPU and can stall CI, so they
        /// are excluded from the light test suite.
        #[cfg(feature = "heavy")]
        mod full_cube {
            use super::*;
            use cubek_reduce::routines::cube::CubeStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Cube(
                        BlueprintStrategy::Inferred(CubeStrategy{ use_planes: false })
                    ),
                },
            );
        }

        #[cfg(feature = "heavy")]
        mod full_cube_plane {
            use super::*;
            use cubek_reduce::routines::cube::CubeStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Cube(
                        BlueprintStrategy::Inferred(CubeStrategy{ use_planes: true })
                    ),
                },
            );
        }

        /// The goal of that test is to limit the size of a cube to `8` to validate multiple cubes
        /// arithmetic.
        ///
        /// With this test, we can't have `use_planes` to true since the `cube_dim.x !=
        /// plane_size`.
        #[cfg(feature = "heavy")]
        mod full_cube_single_plane {
            use super::*;
            use cubek_reduce::{routines::CubeBlueprint, {BoundChecks, IdleMode}};
            use cubecl::prelude::CubeDim;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Cube(
                        BlueprintStrategy::Forced(
                            CubeBlueprint {
                                cube_idle: IdleMode::Terminate,
                                bound_checks: BoundChecks::Mask,
                                num_shared_accumulators: 8,
                                use_planes: false,
                            },
                            CubeDim::new_2d(8, 1),
                        )
                    ),
                },
            );
        }

        /// The goal of that test is to limit the size of a cube to `plane_size` to validate multiple planes
        /// arithmetic.
        mod full_plane_single_plane {
            use super::*;
            use cubek_reduce::{routines::PlaneReduceBlueprint, {BoundChecks, IdleMode}};
            use cubecl::prelude::CubeDim;

            mod plane_size_32 {
                use super::*;

                testgen_reduce!(
                    dtype: $dtype,
                    shape: $shape,
                    strides: $strides,
                    axis: $axis,
                    strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                        vectorization: $vectorization_strategy,
                        routine: RoutineStrategy::Plane(
                            BlueprintStrategy::Forced(
                                PlaneReduceBlueprint {
                                    plane_idle: IdleMode::Terminate,
                                    bound_checks: BoundChecks::Mask,
                                    plane_merge_strategy: PlaneMergeStrategy::Lazy,
                                    plane_dim_ceil: true,
                                },
                                CubeDim::new_2d(32, 2),
                            )
                        ),
                    },
                );
            }

            #[cfg(feature = "extended")]
            mod plane_size_64 {
                use super::*;

                testgen_reduce!(
                    dtype: $dtype,
                    shape: $shape,
                    strides: $strides,
                    axis: $axis,
                    strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                        vectorization: $vectorization_strategy,
                        routine: RoutineStrategy::Plane(
                            BlueprintStrategy::Forced(
                                PlaneReduceBlueprint {
                                    plane_idle: IdleMode::Terminate,
                                    bound_checks: BoundChecks::Mask,
                                    plane_merge_strategy: PlaneMergeStrategy::Lazy,
                                    plane_dim_ceil: true,
                                },
                                CubeDim::new_2d(64, 2),
                            )
                        ),
                    },
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
                strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Plane(
                        BlueprintStrategy::Inferred(PlaneStrategy{ independent: false })
                    ),
                },
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
                strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Plane(
                        BlueprintStrategy::Inferred(PlaneStrategy{ independent: true })
                    ),
                },
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
                strategy: ReduceStrategy { autotune_level: AutotuneLevel::Full,
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Unit(
                        BlueprintStrategy::Inferred(UnitStrategy)
                    ),
                },
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

    // One small point per axis the light suite collapses, so that the halves it
    // drops (`f16`, unvectorized output, a 64-lane plane) stay covered. Under
    // `extended` the full cross product already contains all three.
    #[cfg(not(feature = "extended"))]
    mod collapsed_axis_coverage {
        mod f16_vectorized {
            use cubek_reduce::launch::VectorizationStrategy;

            testgen_reduce!(
                dtype: half::f16,
                shape: shape![4, 8],
                strides: strides![8, 1],
                axis: Some(1),
                vectorization_strategy: VectorizationStrategy {
                    parallel_output_vectorization: true,
                },
            );
        }

        mod f32_unvectorized {
            use cubek_reduce::launch::VectorizationStrategy;

            testgen_reduce!(
                dtype: f32,
                shape: shape![4, 8],
                strides: strides![8, 1],
                axis: Some(1),
                vectorization_strategy: VectorizationStrategy {
                    parallel_output_vectorization: false,
                },
            );
        }

        mod plane_size_64 {
            use cubecl::{config::autotune::AutotuneLevel, prelude::CubeDim};
            use cubek_reduce::{
                BoundChecks, IdleMode, ReduceStrategy,
                launch::{RoutineStrategy, VectorizationStrategy},
                routines::{BlueprintStrategy, PlaneMergeStrategy, PlaneReduceBlueprint},
            };

            testgen_reduce!(
                dtype: f32,
                shape: shape![4, 8],
                strides: strides![8, 1],
                axis: Some(1),
                strategy: ReduceStrategy {
                    autotune_level: AutotuneLevel::Full,
                    vectorization: VectorizationStrategy {
                        parallel_output_vectorization: true,
                    },
                    routine: RoutineStrategy::Plane(
                        BlueprintStrategy::Forced(
                            PlaneReduceBlueprint {
                                plane_idle: IdleMode::Terminate,
                                bound_checks: BoundChecks::Mask,
                                plane_merge_strategy: PlaneMergeStrategy::Lazy,
                                plane_dim_ceil: true,
                            },
                            CubeDim::new_2d(64, 2),
                        )
                    ),
                },
            );
        }
    }
}

mod reduce {
    mod vector_small {
        testgen_reduce!(
            shape: shape![22],
            strides: strides![1],
        );
    }

    mod vector_large {
        testgen_reduce!(
            shape: shape![1024],
            strides: strides![1],
        );
    }

    mod matrix_small {
        testgen_reduce!(
            shape: shape![4, 8],
            strides: strides![8, 1],
        );
    }

    mod matrix_large {
        testgen_reduce!(
            shape: shape![8, 256],
            strides: strides![256, 1],
        );
    }

    mod rank_three_tensor {
        testgen_reduce!(
            shape: shape![16, 16, 16],
            strides: strides![1, 256, 16],
        );
    }

    mod rank_three_tensor_unexact_shape {
        testgen_reduce!(
            shape: shape![11, 12, 13],
            strides: strides![156, 13, 1],
        );
    }

    mod rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![64, 16, 4, 1],
        );
    }

    // TODO not well supported in test-utils
    // mod matrix_with_jumps {
    //     testgen_reduce!(
    //         shape: shape![8, 8],
    //         strides: strides![64, 1],
    //     );
    // }

    mod broadcast_slice_0 {
        testgen_reduce!(
            shape: shape![4, 32],
            strides: strides![0, 1],
        );
    }
}
