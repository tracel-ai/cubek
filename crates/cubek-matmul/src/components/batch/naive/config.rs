use cubecl::{CubeCount, CubeDim};

use crate::{
    components::{batch::BatchConfig, global::memory::GlobalLayoutConfig},
    definition::{CubeCountPlan, MatmulLineSizes, MatmulProblem, MatrixLayout},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NaiveMatmulConfig {}

impl BatchConfig for NaiveMatmulConfig {
    fn cube_dim(&self) -> CubeDim {
        todo!()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        todo!()
    }

    fn can_yield_extra_cubes(&self) -> bool {
        todo!()
    }

    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn cube_count_plan(
        &self,
        _problem: &MatmulProblem,
        _max_cube_count: &CubeCount,
    ) -> CubeCountPlan {
        todo!()
    }
}
