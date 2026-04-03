use cubek_std::MatrixLayout;

use crate::{
    components::{batch::BatchConfig, global::memory::GlobalLayoutConfig},
    definition::{MatmulProblem, MatmulSetupError},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum GemvPlan {
    // Conceptually VecMat
    VecMatDirect,        // execute VecMat as-is
    VecMatTransposeSwap, // execute via transpose-swap (as MatVec)

    // Conceptually MatVec
    MatVecDirect,        // execute MatVec as-is
    MatVecTransposeSwap, // execute via transpose-swap (as VecMat)
}

impl GemvPlan {
    pub(crate) fn from_problem(problem: &MatmulProblem) -> Result<GemvPlan, MatmulSetupError> {
        if problem.m == 1 {
            Ok(match problem.rhs_layout {
                MatrixLayout::ColMajor => GemvPlan::VecMatDirect,
                MatrixLayout::RowMajor => GemvPlan::VecMatTransposeSwap,
            })
        } else if problem.n == 1 {
            Ok(match problem.lhs_layout {
                MatrixLayout::ColMajor => GemvPlan::MatVecTransposeSwap,
                MatrixLayout::RowMajor => GemvPlan::MatVecDirect,
            })
        } else {
            Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Problem is not a valid GEMV, got (m,n,k)=({:?},{:?},{:?})",
                problem.m, problem.n, problem.k
            ))))
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct VecMatPlaneParallelConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) plan: GemvPlan,
}

impl BatchConfig for VecMatPlaneParallelConfig {
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
}
