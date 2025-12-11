#[cfg(not(feature = "matmul_tests_hypercube"))]
pub mod row_fp {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::RowMajor))
            .cube_count_plan(CubeCountPlanSelection::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlecol_fp {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleColMajor(2)))
            .cube_count_plan(CubeCountPlanSelection::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod col_fl {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::ColMajor))
            .cube_count_plan(CubeCountPlanSelection::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod swizzlerow_fl {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleRowMajor(2)))
            .cube_count_plan(CubeCountPlanSelection::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

#[cfg(feature = "matmul_tests_hypercube")]
mod row_sm_exact {
    use super::*;
    use cubek_matmul::components::batch::{
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::RowMajor))
            .cube_count_plan(CubeCountPlanSelection::Sm {
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
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleColMajor(2)))
            .cube_count_plan(CubeCountPlanSelection::Sm {
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
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::RowMajor))
            .cube_count_plan(CubeCountPlanSelection::Sm {
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
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection, SmAllocation,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleRowMajor(2)))
            .cube_count_plan(CubeCountPlanSelection::Sm {
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
        CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
    };

    fn hypercube_selection(tiling_scheme: &TilingScheme) -> HypercubeSelection {
        HypercubeSelection::builder(tiling_scheme)
            .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleRowMajor(2)))
            .cube_count_plan(CubeCountPlanSelection::Spread)
            .build()
    }

    include!("partition_buffering.rs");
}
