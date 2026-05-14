use cubek_std::MatrixLayout;

use crate::{
    components::{
        batch::{BatchConfig, CheckBounds},
        global::memory::GlobalLayoutConfig,
    },
    definition::{MatmulProblem, MatmulSetupError},
};

/// Selects which kernel variant the plane-parallel matmul should execute.
///
/// `Standard` is regular GEMM (`m, n >= 1`). The other variants are GEMV cases
/// where one of the two outer dimensions collapses to 1 and the operand stored
/// with K on the slow stride needs a staging tile.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum GemmPlan {
    /// Regular GEMM: lhs Row × rhs Col → out Row. Planes parallelize along N.
    Standard,
    /// GEMV with `m = 1`, rhs ColMajor. Planes parallelize along N.
    VecMatColMajor,
    /// GEMV with `m = 1`, rhs RowMajor. Uses the transposed (staged) kernel.
    VecMatRowMajor,
    /// GEMV with `n = 1`, lhs RowMajor. Planes parallelize along M.
    MatVecRowMajor,
    /// GEMV with `n = 1`, lhs ColMajor. Uses the transposed (staged) kernel.
    MatVecColMajor,
}

impl GemmPlan {
    /// Infers the GEMV plan when the problem is degenerate (`m == 1` or
    /// `n == 1`). Returns an error for full GEMM problems — callers that want
    /// the standard plan must select [`GemmPlan::Standard`] explicitly.
    pub(crate) fn gemv_from_problem(problem: &MatmulProblem) -> Result<GemmPlan, MatmulSetupError> {
        if problem.m == 1 {
            Ok(match problem.rhs_layout {
                MatrixLayout::ColMajor => GemmPlan::VecMatColMajor,
                MatrixLayout::RowMajor => GemmPlan::VecMatRowMajor,
            })
        } else if problem.n == 1 {
            Ok(match problem.lhs_layout {
                MatrixLayout::ColMajor => GemmPlan::MatVecColMajor,
                MatrixLayout::RowMajor => GemmPlan::MatVecRowMajor,
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
pub struct GemmPlaneParallelConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) plan: GemmPlan,
    pub(crate) check_bounds: CheckBounds,
}

impl BatchConfig for GemmPlaneParallelConfig {
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
