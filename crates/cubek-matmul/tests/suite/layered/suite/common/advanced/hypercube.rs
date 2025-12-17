#[cfg(not(feature = "matmul_tests_hypercube"))]
pub mod row_fp {
    use super::*;
    use cubek_matmul::definition::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::RowMajor))
            .cube_count_plan(CubeCountPlanBlueprint::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlecol_fp {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::SwizzleColMajor(2)))
            .cube_count_plan(CubeCountPlanBlueprint::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod col_fl {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::ColMajor))
            .cube_count_plan(CubeCountPlanBlueprint::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlerow_fl {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::SwizzleRowMajor(2)))
            .cube_count_plan(CubeCountPlanBlueprint::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod row_sm_exact {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::RowMajor))
            .cube_count_plan(CubeCountPlanBlueprint::Sm {
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
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::SwizzleColMajor(2)))
            .cube_count_plan(CubeCountPlanBlueprint::Sm {
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
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::RowMajor))
            .cube_count_plan(CubeCountPlanBlueprint::Sm {
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
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::SwizzleRowMajor(2)))
            .cube_count_plan(CubeCountPlanBlueprint::Sm {
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
    use cubek_matmul::components::batch::{
        CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint, HypercubeBlueprint,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeBlueprint {
        HypercubeBlueprint::builder(tiling_scheme)
            .global_order(GlobalOrderBlueprint::Fixed(GlobalOrder::SwizzleRowMajor(2)))
            .cube_count_plan(CubeCountPlanBlueprint::Spread)
            .build()
    }

    include!("partition_buffering.rs");
}
