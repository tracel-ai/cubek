use cubek_std::MatrixLayout;

use crate::{
    components::{
        batch::{BatchConfig, CheckBounds},
        global::memory::GlobalLayoutConfig,
    },
    definition::{MatmulProblem, MatmulSetupError},
};

/// Role and layout of a single operand for the plane-parallel kernel.
///
/// An operand is either a matrix (with a memory layout) or a vector, when
/// its non-k dimension is 1.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum OperandKind {
    RowMajor,
    ColMajor,
    Vector,
}

/// Shape + layout of the matmul problem the plane-parallel kernel will run.
///
/// Composed from the per-side [`OperandKind`]s; the dispatch path (simple
/// vs staged) is derived from their combination.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MatmulKind {
    pub lhs: OperandKind,
    pub rhs: OperandKind,
}

/// Which output axis the planes within a cube enumerate.
///
/// Exactly one of `(planes_per_m, planes_per_n)` equals `num_planes`; the
/// other is 1. For vec/mat the choice is forced (planes split the non-1
/// axis); for mat/mat it's an algorithmic choice made by the routine.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlanesSplit {
    M,
    N,
}

/// How the kernel body is dispatched.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum DispatchPath {
    /// Plane reduces over k for one `(m, n)` output cell.
    Simple,
    /// Staged tile reduction; the matrix's k axis is non-contiguous and
    /// must be reloaded through shared memory. Vec is lhs, mat is rhs RowMajor.
    StagedVecMat,
    /// Staged tile reduction; the matrix's k axis is non-contiguous and
    /// must be reloaded through shared memory. Mat is lhs ColMajor, vec is rhs.
    StagedMatVec,
}

impl MatmulKind {
    /// Infer the kind from problem shape and operand layouts.
    pub(crate) fn from_problem(problem: &MatmulProblem) -> Result<MatmulKind, MatmulSetupError> {
        let lhs = operand_kind(problem.m, problem.lhs_layout);
        let rhs = operand_kind(problem.n, problem.rhs_layout);
        let kind = MatmulKind { lhs, rhs };
        kind.validate_layouts(problem)?;
        Ok(kind)
    }

    fn validate_layouts(self, problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        // Mat × Mat is only supported with lhs RowMajor + rhs ColMajor.
        let mat_mat = !matches!(self.lhs, OperandKind::Vector)
            && !matches!(self.rhs, OperandKind::Vector);
        if mat_mat
            && !matches!(
                (self.lhs, self.rhs),
                (OperandKind::RowMajor, OperandKind::ColMajor)
            )
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Plane-parallel GEMM requires lhs RowMajor and rhs ColMajor, got lhs={:?}, rhs={:?}",
                problem.lhs_layout, problem.rhs_layout
            ))));
        }
        Ok(())
    }

    /// Which kernel body should run for this kind.
    pub fn dispatch_path(self) -> DispatchPath {
        match (self.lhs, self.rhs) {
            (OperandKind::Vector, OperandKind::RowMajor) => DispatchPath::StagedVecMat,
            (OperandKind::ColMajor, OperandKind::Vector) => DispatchPath::StagedMatVec,
            _ => DispatchPath::Simple,
        }
    }
}

fn operand_kind(dim: usize, layout: MatrixLayout) -> OperandKind {
    if dim == 1 {
        OperandKind::Vector
    } else {
        match layout {
            MatrixLayout::RowMajor => OperandKind::RowMajor,
            MatrixLayout::ColMajor => OperandKind::ColMajor,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GemmPlaneParallelConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) kind: MatmulKind,
    pub(crate) planes_split: PlanesSplit,
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
