use cubek_std::MatrixLayout;

use crate::{
    components::{
        batch::{BatchConfig, CheckBounds},
        global::memory::GlobalLayoutConfig,
    },
    definition::{MatmulProblem, MatmulSetupError},
};

/// Shape + layout of the matmul problem the plane-parallel kernel will run.
///
/// The plane-parallel family treats GEMV as a degenerate GEMM where one of
/// `m, n` collapses to 1; each variant pins the relevant operand layouts.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum MatmulKind {
    /// Full GEMM with lhs RowMajor and rhs ColMajor.
    MatRowMatCol,
    /// `m = 1`, rhs ColMajor.
    VecMatCol,
    /// `m = 1`, rhs RowMajor. Uses the staged (transposed) kernel.
    VecMatRow,
    /// `n = 1`, lhs RowMajor.
    MatRowVec,
    /// `n = 1`, lhs ColMajor. Uses the staged (transposed) kernel.
    MatColVec,
}

impl MatmulKind {
    /// Infer the kind from problem shape and operand layouts.
    pub(crate) fn from_problem(problem: &MatmulProblem) -> Result<MatmulKind, MatmulSetupError> {
        if problem.m == 1 && problem.n == 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Plane-parallel matmul requires at least one of m, n > 1".to_string(),
            )));
        }

        if problem.m == 1 {
            return Ok(match problem.rhs_layout {
                MatrixLayout::ColMajor => MatmulKind::VecMatCol,
                MatrixLayout::RowMajor => MatmulKind::VecMatRow,
            });
        }
        if problem.n == 1 {
            return Ok(match problem.lhs_layout {
                MatrixLayout::ColMajor => MatmulKind::MatColVec,
                MatrixLayout::RowMajor => MatmulKind::MatRowVec,
            });
        }
        match (problem.lhs_layout, problem.rhs_layout) {
            (MatrixLayout::RowMajor, MatrixLayout::ColMajor) => Ok(MatmulKind::MatRowMatCol),
            (lhs, rhs) => Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Plane-parallel GEMM requires lhs RowMajor and rhs ColMajor, got lhs={:?}, rhs={:?}",
                lhs, rhs
            )))),
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GemmPlaneParallelConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) kind: MatmulKind,
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
