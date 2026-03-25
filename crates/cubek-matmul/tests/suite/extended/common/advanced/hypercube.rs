#[cfg(not(feature = "matmul_tests_hypercube"))]
pub mod row_fp {
    use super::*;
    use cubek_matmul::definition::{MatmulProblem, TilingScheme};
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlecol_fp {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleCol(2))
            .cube_count_strategy(CubeCountStrategy::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod col_fl {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::ColMajor)
            .cube_count_strategy(CubeCountStrategy::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlerow_fl {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod row_sm_exact {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Exact,
                cubes_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlecol_sm_exact {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::SwizzleColMajor(2)))
            .global_order_strategy()
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Exact,
                cubes_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod row_sm_full {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Full,
                cubes_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlerow_cube_full {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Full,
                cubes_first: true,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlerow_spread {
    use super::*;
    use cubek_matmul::definition::TilingScheme;
    use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Spread)
            .build()
    }

    include!("partition_buffering.rs");
}
